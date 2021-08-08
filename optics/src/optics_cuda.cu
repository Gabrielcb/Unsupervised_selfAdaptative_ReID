#include "../include/grafo.h"

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


__global__ void montandoVa_n_cuda(float *X, int *Va_n, float eps, int how_large, int numDim) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	float dist;

	if(idx < how_large){
		Va_n[idx] = 0;
		for(i = 0; i < how_large; i++){
			dist = 0;
			if(idx != i){
				for(j = 0; j < numDim; j++)
					dist += (float) ((X[idx * numDim + j] - X[i * numDim + j]) * (X[idx * numDim + j] - X[i * numDim + j]));
				
				if(dist <=  eps * eps){
					Va_n[idx] += 1;
				}
			}
		}
	}
}


__global__ void montandoEa_cuda(float *X, int *Va_i, int *Va_n, int *Ea_ids, float *Ea_dist, float eps, int how_large, int numDim) {

	int i, j;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pointer = Va_i[idx];
	int numVizinhos = Va_n[idx];
	float dist;
	
	if(idx < how_large && numVizinhos > 0){
		for(i = 0; i < how_large; i++){
			dist = 0;
			if(idx != i){
				for(j = 0; j < numDim; j++)
					dist += ((X[idx * numDim + j] - X[i * numDim + j]) * (X[idx * numDim + j] - X[i * numDim + j]));

				if(dist <=  eps * eps) {
					Ea_ids[pointer] = i;
					Ea_dist[pointer] = sqrtf(dist);
					pointer++;
				}
			}
		}
	}

}

__global__ void ordenandoEA(int *Va_i, int *Va_n, int *Ea_ids, float *Ea_dist, int how_large){

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i,j,min;
	int aux_id;
	float aux_dist;
	int numVizinhos = Va_n[idx];
	int inicio = Va_i[idx];
	int fim = (inicio + numVizinhos) -1;
	
	if(idx < how_large && numVizinhos > 0){
		for (i = inicio; i <= fim; i++) {
			min = i;
			for (j = (i+1); j <= fim; j++) {
				if(Ea_dist[j] < Ea_dist[min]) 
					min = j;
			}
			if (i != min) {
				aux_dist = Ea_dist[i];		
				aux_id = Ea_ids[i];
				Ea_dist[i] = Ea_dist[min];
				Ea_ids[i] = Ea_ids[min];
				Ea_dist[min] = aux_dist;
				Ea_ids[min] = aux_id;
			}
		}
	}
}


extern "C" {

	int ordenaCPU (const void *a, const void *b) {
		itemEa *x = (itemEa *) a;
		itemEa *y = (itemEa *) b;
		if (x->distancia > y->distancia) return 1;
		if (x->distancia < y->distancia) return -1;

		return 0;
	}

	double tempoAtual2() {
		struct timeval tv2;
		
		gettimeofday(&tv2,0);
		return tv2.tv_sec + tv2.tv_usec/1.e6;
	}
	
	void ordenandoEA(int *Va_i, int *Va_n, int *Ea_ids, float *Ea_dist, int how_large, int idx){

		int i,j,min;
		int aux_id;
		float aux_dist;
		int numVizinhos = Va_n[idx];
		int inicio = Va_i[idx];
		int fim = (inicio + numVizinhos) -1;
		
		
		if(idx < how_large && numVizinhos > 0){
			for (i = inicio; i <= fim; i++) {
				min = i;
				for (j = (i+1); j <= fim; j++) {
					if(Ea_dist[j] < Ea_dist[min]) 
					min = j;
				} if (i != min) {
					aux_dist = Ea_dist[i];
					aux_id = Ea_ids[i];
					Ea_dist[i] = Ea_dist[min];
					Ea_ids[i] = Ea_ids[min];
					Ea_dist[min] = aux_dist;
					Ea_ids[min] = aux_id;
				}
			}
		}
	}
	

	void montandoVa_n_cpup(point *L, int *Va_n, float eps, int how_large, int idx) {

		int i, j = 0;
		float de;

		Va_n[idx] = 0;
		for(i = 0; i < how_large; ++i) {
			de = 0;
			if(idx != i) {
				for(j = 0; j < numDim; ++j){
					de += (float) ((L[idx].X[j] - L[i].X[j]) * (L[idx].X[j] - L[i].X[j]));
				}


				if( de <=  eps * eps){
					Va_n[idx] += 1;
				}
			}
		}
	}


	itemEa* montaGrafoOrdenaGPU(point *L, int *Va_i, int *Va_n, float eps, int c){
		double tmi, tmf, toi, tof;
		double tmt = 0.0;
		double tot = 0.0;
		int i, j;
		
		tmi = tempoAtual2();
		
		int *Ea_ids = NULL; 	// Esse vetor armazena, para cada ponto, todos as ids de seus vizinhos.
		float *Ea_dist = NULL; 	// Esse vetor armazena, para cada ponto, todas as distancias até seus vizinhos.
		itemEa *Ea = NULL;
		
		float *X = NULL;

		X = (float*) calloc(how_large * numDim, sizeof(float));
		float eps_f = (float) eps;


		for(i = 0; i < how_large; ++i) {
			for(j = 0; j < numDim; ++j){
				X[i * numDim + j] = L[i].X[j];
			}
		}


		//Passando X para device
		float *X_D;
		cudaMalloc((void **) &X_D, sizeof(float) * how_large * numDim);
		cudaMemcpy(X_D, X, sizeof(float) * how_large * numDim, cudaMemcpyHostToDevice);
		free(X);

		//Passando Va_n para device
		int *Va_n_D;
		cudaMalloc ((void **) &Va_n_D, sizeof(int)*how_large);

		//Passando Va_i para device
		int *Va_i_D ;
		cudaMalloc ((void **) &Va_i_D, sizeof(int)*how_large);
	
		dim3 grid, block;

		block.x = 512;
		grid.x = ((how_large) + block.x-1) / block.x;
		montandoVa_n_cuda<<<grid, block, 0>>>(X_D, Va_n_D, eps_f, how_large, numDim);
		cudaDeviceSynchronize();
		
		//Voltado Va_n_D pro Host
		cudaMemcpy(Va_n, Va_n_D, sizeof(int) * how_large, cudaMemcpyDeviceToHost);

		thrust::device_ptr<int> dptr_indAresta(Va_i_D);
		thrust::device_ptr<int> dptr_grauVert(Va_n_D);
		thrust::exclusive_scan(dptr_grauVert, dptr_grauVert + how_large, dptr_indAresta);
		cudaDeviceSynchronize();


		// =================================================================
		int tGr, tIn;
		cudaMemcpy(&tIn, &Va_i_D[how_large - 1], sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tGr, &Va_n_D[how_large - 1], sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Va_i, Va_i_D, sizeof(int) * how_large, cudaMemcpyDeviceToHost);
		numEdges = tGr + tIn;


		Ea_ids = (int*) malloc((sizeof(int) * numEdges));
		Ea_dist = (float*) malloc((sizeof(float) * numEdges));
		Ea = (itemEa*) malloc((sizeof(itemEa) * numEdges));
		// ========================================================================================	



	  	//Passando Ea_ids_D e Ea_dist_D para device
		int *Ea_ids_D = NULL;
		cudaMalloc ((void **) &Ea_ids_D, sizeof(int) * numEdges);
		
		float *Ea_dist_D = NULL;	
		cudaMalloc ((void **) &Ea_dist_D, sizeof(float) * numEdges);


		montandoEa_cuda<<<grid, block, 0>>>(X_D, Va_i_D, Va_n_D, Ea_ids_D, Ea_dist_D, eps_f, how_large, numDim);
		cudaDeviceSynchronize();

		cudaFree(X_D);
					
		tmf = tempoAtual2();
		tmt = tmf - tmi;
		toi = tempoAtual2();
		//ordenandoEA<<<grid, block, 0>>>(Va_i_D, Va_n_D, Ea_ids_D, Ea_dist_D, how_large);
		
		//Voltando Ea_ids e Ea_dist pra host
		cudaMemcpy(Ea_ids, Ea_ids_D, sizeof (int) * numEdges, cudaMemcpyDeviceToHost);
		cudaMemcpy(Ea_dist, Ea_dist_D, sizeof (float) * numEdges, cudaMemcpyDeviceToHost);
	
					
		for(i = 0; i < numEdges; ++i){
			Ea[i].id = Ea_ids[i];
			Ea[i].distancia = Ea_dist[i];
		}
		
		tof = tempoAtual2();
		tot = tof - toi;
		
		cudaFree(Va_i_D);
		cudaFree(Va_n_D);
		cudaFree(Ea_ids_D);
		cudaFree(Ea_dist_D);

		
		
		printf("GPU MONTAGEM\t%lf\n",tmt);
		printf("GPU ORDENACAO\t%lf\n",tot);
	
		return Ea;
	}
	
	
	itemEa* montaGrafoOrdenaCPUParalela(point *L, int *Va_i, int *Va_n, float eps, int c){

		double tmi, tmf, toi, tof;
		double tmt = 0.0;
		double tot = 0.0;
		
		tmi = tempoAtual2();
		
		itemEa *Ea;
		
		int i;

		#pragma omp parallel for schedule(dynamic)
		for(i=0; i<how_large; i++){
			montandoVa_n_cpup(L, Va_n, eps, how_large, i);
		}
		
		Va_i[0] = 0;
		numEdges = Va_n[0];
		for(i=1; i<how_large; i++){
			Va_i[i] = Va_i[i-1] + Va_n[i];
			numEdges += Va_n[i];
		}
			
		Ea = (itemEa*) malloc ((sizeof(itemEa)*numEdges));
		
		//MONTANDO EA
		#pragma omp parallel for schedule(dynamic)
		for(i=0; i < how_large; ++i){
			int idx = i;
			int j, k;
			float de;
			int pointer = Va_i[idx];
			int numVizinhos = Va_n[idx];
			
			if(numVizinhos > 0){
				for(j = 0; j < how_large; ++j){
					de = 0;
					if(idx != j){
						for(k = 0; k < numDim; ++k){
							de += (float) ((L[idx].X[k] - L[i].X[k]) * (L[idx].X[k] - L[i].X[k]));
						}

						if( de <=  eps*eps) {
							Ea[pointer].id = j;
							Ea[pointer].distancia = sqrt(de);
							pointer++;
						}
					}
				}
			}
		}
					
		tmf = tempoAtual2();
		tmt = tmf - tmi;
		toi = tempoAtual2();	
		
		
		//Ordena o vetor de adjacentes
		//#pragma omp parallel for schedule(dynamic)
		for(i = 0; i < how_large; ++i) {
			int num_vizinhos = Va_n[i];
			int inicio_listaVizinhos = Va_i[i];	
			qsort((void*) &Ea[inicio_listaVizinhos], num_vizinhos, sizeof(itemEa), ordenaCPU);
		}
		
		tof = tempoAtual2();
		
		tot = tof - toi;
				
		
		printf("CPUN MONTAGEM\t%lf\n",tmt);
		printf("CPUN ORDENACAO\t%lf\n",tot);
	
		return Ea;
	}
}
