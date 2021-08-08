#ifndef OPTICS_C
#define OPTICS_C

#include "../include/optics.h"

void expandClusterOrder(point *points, point *current, int minPts, float radius, itemEa *Ea, int *Va_i, int *Va_n, PriorityQueue *heap){
	int i, j = 0;

	current->processed = 1;
	
	// Define the core distance of the current point. If he cant be core, his coreDist atribute will be UNDF.
	setCoreDist(current, minPts, Ea, Va_i, Va_n);

	// Escrever no arquivo os dados do ponto alcançado
	for(i = 0; i < numDim; ++i)
		fprintf(escrever, "%.10f ", current->X[i]);
	fprintf(escrever, "%.10f %.10f\n", current->coreDist, current->reachDist);

	if(current->coreDist != UNDF){
		orderSeedsUpdate(points, current, radius, Ea, Va_i, Va_n, heap);
	}
	
	while(!heapIsEmpty(heap)){
		current = getNextHeap(heap);
		current->processed = 1;
		setCoreDist(current, minPts, Ea, Va_i, Va_n);
		
		// Escrever no arquivo os dados do ponto alcançado
		for(i = 0; i < numDim; ++i)
			fprintf(escrever, "%.10f ", current->X[i]);
		fprintf(escrever, "%.10f %.10f\n", current->coreDist, current->reachDist);
		
		if(current->coreDist != UNDF){
			orderSeedsUpdate(points, current, radius, Ea, Va_i, Va_n, heap);
		}	
	}
}

void setCoreDist(point *current, int minPts, itemEa *Ea, int *Va_i, int *Va_n){
	// Se o ponto tiver vizinhos o suficiente pra ser core determina o valor de coreDist pra ele
	// Caso contrário a coreDist fica como indefinida


	
	if ((Va_n[current->id] >=  minPts - 1) && (minPts <= how_large)) {

		// AONDE ELE COMEÇA, A QUANTIDADE DE PONTOS A SEREM ORDENADOS, O TAMANHO DOS PONTOS E A FUNÇAO DE COMPARAÇÃO
		qsort(&(Ea[Va_i[current->id]]), Va_n[current->id], sizeof(itemEa), ordena);
		current->coreDist = Ea[Va_i[current->id] + minPts - 2].distancia; // -2 em Ea porque esse vetor não conta ele mesmo como vizinho
	} else {
		current->coreDist = UNDF;
	}
}

void orderSeedsUpdate(point *points, point *o, float radius, itemEa *Ea, int *Va_i, int *Va_n, PriorityQueue *heap){
	int i = 0;
	float cdist, newrdist;

	point *p;
	cdist = o->coreDist;


	// Do the process below to the neighbors of "o"
	for(i = Va_i[o->id]; (i < Va_i[o->id] + Va_n[o->id]) && (Va_n[o->id] > 0); ++i){ 
		p = &points[Ea[i].id];
		if(!p->processed){

			newrdist = max(cdist, Ea[i].distancia);
			
			// Verifica se o ponto já está na heap
			if(p->reachDist == UNDF) {
				// Caso NÃO esteja na heap, define-se a reachDist dele e insira-o na heap
				p->reachDist = newrdist;
				insertHeap(heap, p);
			} else {
				if(newrdist < p->reachDist){
					// Caso esteja na heap, atualiza-se sua reachDist e a heap
					p->reachDist = newrdist;
					promoteElement(heap, p->pqPos);
				}
			}
		}
	}
}

#endif