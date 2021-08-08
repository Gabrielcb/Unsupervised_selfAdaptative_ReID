#include "../include/grafo.h"

// =====================================================================================================================
// ============================================== Funções de conveniência ==============================================
point* calloc_arrayOfPoints(int size) {
	point *vetor = NULL;
	vetor = calloc (size, sizeof(point));

	if(vetor == NULL) {
		printf("Falha em 'calloc_arrayOfPoints'.\n");
		exit(EXIT_FAILURE);
	}
	
    return vetor;
}
// ==================



// ==================
itemEa* calloc_edgeArray(int size) {
	
	itemEa *vetor = NULL;
	vetor = calloc (size, sizeof(itemEa));

	if(vetor == NULL) {
		printf("Falha em 'calloc_edgeArray'.\n");
		exit(EXIT_FAILURE);
	}
	
    return vetor;
}
// ==================



// ==================
float euclideanDistance(point P, point P_l) {
	int i;
	float tmp = 0;
	for(i = 0; i < numDim; ++i){
		tmp += (P.X[i] - P_l.X[i]) * (P.X[i] - P_l.X[i]); 
	}
	return sqrtf(tmp);
}
// ==================



// ==================
float max(const float a, const float b){
	if(a >= b) return a;
	else return b;
}
// ==================



// ==================
int ordena (const void *a, const void *b) {
	itemEa *x = (itemEa *) a;
	itemEa *y = (itemEa *) b;
	if (x->distancia > y->distancia) return 1;
	if (x->distancia < y->distancia) return -1;
	return 0;
}
// ==================



// ==================
int readFile(const char *name, point *list, float eps) {
	FILE *file;
	int i = 0, j = 0, k = 0;

	// Verifica se o arquivo existe 
	file = fopen(name, "r");
	if (file == NULL) {
		printf("\t-> O arquivo %s não foi encontrado.\n", name);
		exit(0);
	}
	
	int dim = 0, tmp = 0;
	for(i = 0; ;++i){
		for(dim = 0; dim < numDim; ++dim)
			if(fscanf(file, "%f", &list[i].X[dim]) == EOF){
				tmp = EOF;
				break;
			}
		if(tmp == EOF)
			break;
		
		list[i].id = i;
		list[i].reachDist = UNDF;
		list[i].coreDist = 0;
		list[i].processed = 0;
		list[i].pqPos = -1;
	}

	// Tempo Montagem Inicial
	tmi = currentTime();

	// THIS SUB ROUTINE IS ONLY FOR THE SEQUENTIAL EXECUTION 
	if(cpu1){
		for(j = 0; j < i; j++){	
			for(k = j; k < i; k++){
				if (euclideanDistance(list[j], list[k]) <= eps) {
					numEdges += 2;
				}
			}
		}
	}
	// Tempo Montagem Final
	tmf = currentTime();
	// Tempo Montagem Total
	tmt += tmf - tmi;
	
	fclose(file);
	
	// 'i' representa o número de pontos no arquivo de texto.
	return i;
}
// ==================



// ==================
void temposExecucao(double *utime, double *stime, double *total_time) {
    int rc;
    
    if((rc = getrusage(RUSAGE_SELF, &resources)) != 0)
       perror("getrusage Falhou");

    *utime = (double) resources.ru_utime.tv_sec + (double) resources.ru_utime.tv_usec * 1.e-6 ;
    *stime = (double) resources.ru_stime.tv_sec + (double) resources.ru_stime.tv_usec * 1.e-6 ;
    *total_time = *utime + *stime;
}
// ==================



// ==================
double currentTime() {
	gettimeofday(&tv,0);
	return tv.tv_sec + tv.tv_usec/1.e6;
}
// ==================





// =====================================================================================================================
// ================================================= Funções do Grafo ==================================================
void montaGrafoOrdenaCPU(point *L, int *Va_i,int *Va_n, itemEa *Ea, float eps, int c) {
	int i = 0, j = 0;
	int auxEa = 0;
	float distancia = 0;
	
	int num_vizinhos = 0;
	int inicio_listaVizinhos = 0;
	
	for(i = 0; i < how_large; i++){	
		tmi = currentTime();
		
		Va_i[i] = auxEa;
		Va_n[i] = 0;
		for(j = 0; j < how_large; ++j) {

			if(i != j){
				distancia = euclideanDistance(L[i],L[j]);
				if(distancia <= eps){
					Ea[auxEa].id = j;
					Ea[auxEa].distancia = distancia;
					auxEa += 1;
					Va_n[i] += 1;
				}
			}
		}
		
		tmf = currentTime();
		tmt += tmf -tmi;		
		toi = currentTime();
		
		// Ordena o vetor de adjacentes
		num_vizinhos = Va_n[i];
		inicio_listaVizinhos = Va_i[i];	
		qsort((void*) &Ea[inicio_listaVizinhos], num_vizinhos, sizeof(itemEa), ordena);
		
		tof = currentTime();
		tot += tof - toi;

	}
	
	printf("CPU MONTAGEM\t%lf\n", tmt);
	printf("CPU ORDENACAO\t%lf\n", tot);
	
}
// ==================





// =====================================================================================================================
// ================================================= OPERAÇÕES DA HEAP =================================================

PriorityQueue* createHeap(int size){
	PriorityQueue *heap;
	heap = malloc(sizeof(PriorityQueue));
	heap->pq = calloc(size, sizeof(element));
	// O número de elementos na heap é exatamente o mostrado, não há nessecidade de subtrair 1 por que
	// vetor em C começa a partir do zero.
	if (heap != NULL) heap->size = 0;
	else printf("Error in createHeap function\n");
	return heap;
 }

void destroyHeap(PriorityQueue *heap){
	free(heap);
}

int heapSize(PriorityQueue *heap){
	if(heap == NULL){
		printf("Error in heapSize function\n");
		return -1;
	}
	else return(heap->size);
}

int heapIsFull(PriorityQueue *heap){
	if(heap == NULL){
		printf("Error in heapIsFull function\n");
		return -1;
	}
	else return (heap->size == how_large);
}

int heapIsEmpty(PriorityQueue *heap){
	if(heap == NULL){
		printf("Error in heapIsEmpty function\n");
		return -1;
	}
	else return (heap->size == 0);
}

int insertHeap(PriorityQueue *heap, point *p){
	if(heap == NULL){
		printf("InsertHeap fail.\n");
		return 0;
	}
	if(heapIsFull(heap)){
		printf("Heap is full, cannot insert anymore elements\n");
		return 0;
	}
	// Insert in last position of the array
	heap->pq[heap->size].p = p;
	heap->pq[heap->size].p->pqPos = heap->size;
	// heap->pq[heap->size].priority = p->reachDist;

	// Then promote him
	promoteElement(heap, heap->size);
	heap->size++;
	return 1;
}

void promoteElement(PriorityQueue *heap, int son){
	int dad;
	element temp;
	// Locate the index of the dad node.
	dad = (son - 1) / 2;

	// While the priority of the dad node is bigger than the son priority,
	// change ther places
	while((son > 0) && (heap->pq[dad].p->reachDist > heap->pq[son].p->reachDist)){
	//while((son > 0) && (heap->pq[dad].priority >= heap->pq[son].priority)){

		temp = heap->pq[son];
		heap->pq[son] = heap->pq[dad];
		heap->pq[dad] = temp;

		// Update positions
		heap->pq[son].p->pqPos = son;
		heap->pq[dad].p->pqPos = dad;

		son = dad;
		dad = (dad - 1) / 2;
		
	}

	if(son != heap->pq[son].p->pqPos) {
		printf("ERROR, ERROR, ERROR na ordenação da heap(promoteElement)!\n");
		exit(EXIT_FAILURE);
	}
}

point* getNextHeap(PriorityQueue *heap){
	if(heap == NULL){
		printf("getNextHeap fail.\n");
		return;
	}

	point *temp;

	// Copies the first element to temp;
	temp = heap->pq[0].p;

	// Overwrite the last element of the heap to the first position 
	heap->pq[0] = heap->pq[heap->size - 1];
	heap->pq[0].p->pqPos = 0;
	
	// Decrement its size
	heap->size--;

	// Then demote the first element to the position he should be according to his
	// Priority
	demoteElement(heap, 0);

	temp->pqPos = -1;
	return temp;
}

void demoteElement(PriorityQueue *heap, int dad){

	element temp;
	int son = 2 * dad + 1;

	// Update the priority of the son node in case this function is called on its own (outside insertHeap)
	//heap->pq[dad].priority = heap->pq[dad].p->reachDist;

	while(son < heap->size){
		// Dad have 2 sons ? Wich one is the smaller (higher priority)
		if(son < heap->size - 1)
			//if(heap->pq[son].priority > heap->pq[son + 1].priority)
			if(heap->pq[son].p->reachDist > heap->pq[son + 1].p->reachDist)
				son++;
		
		// Does the dad have bigger priority than their son
		if(heap->pq[dad].p->reachDist <= heap->pq[son].p->reachDist)
			break;

		// Changes dad and son of places
		temp = heap->pq[dad];
		heap->pq[dad] = heap->pq[son];
		heap->pq[son] = temp;

		// Update positions
		heap->pq[son].p->pqPos = son;
		heap->pq[dad].p->pqPos = dad;

		dad = son;
		son = 2 * dad + 1;
		
	}

	if(dad != heap->pq[dad].p->pqPos) {
		printf("ERROR, ERROR, ERROR na ordenação da heap(demoteElement)!\n");
		exit(EXIT_FAILURE);
	}
}

int compare(const void *a, const void *b, int index) {
	point *element1 = (point *) a;
	point *element2 = (point *) b;
	if (element1->X[index] > element2->X[index]) return 1;
	if (element1->X[index] < element2->X[index]) return -1;
	return 0;
}




// ================================================================================

void teste(float *X, int *Va_n, float eps, int how_large, int numDim, int idx) {

	int i, j;
	float dist;

	if(idx < how_large){
		Va_n[idx] = 0;
		for(i = 0; i < how_large; i++){
			dist = 0;
			if(idx != i){
				for(j = 0; j < numDim; j++)
					dist += (float) ((X[idx + j] - X[i * numDim + j]) * (X[idx + j] - X[i * numDim + j]));
				
				if(dist <=  eps * eps){
					printf("\tidx: %d\n\tdistTo: %d\n\tdist: %.2f\n\n", idx, i, dist);
					Va_n[idx] += 1;
				}
			}
		}
	}
}