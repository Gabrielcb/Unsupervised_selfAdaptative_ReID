#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include <omp.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define UNDF 999999.0

#ifndef LISTA_H
#define LISTA_H


typedef struct {
    int id;
    float distancia;
}itemEa;


int numDim;

typedef struct {
    float *X;          // Dados do ponto
  
    int id;
    float coreDist;    // Minimum distance that makes this point a core. If it does not exist, set to UNDF.
    float reachDist;   // Distance that indicates the reachability of this points from other points in the same cluster. UNDF in case it does not exist.
    int processed;     // Indicate if the point was already or not processed. 1 for TRUE and 0 for FALSE.
    int pqPos;         // Position of the point in the priority queue.
}point;

int numEdges;
int how_large;

// Variáveis para guardar o tempo
double tmi, tmf, toi, tof;
double tmt;
double tot;


// Variáveis usadas para armazenar o tempo consumido pela execução do programa.
struct rusage resources;
struct rusage ru;
struct timeval tim;
struct timeval tv;


// Variáveis para definir em qual PU(Processing Unit) irá ocorrer o processamento do grafo.
int cuda;
int cpu1;
int cpuN;


// Variável para escrever em um arquivo nomeado 'pontosOrdenados' o output do Optics.
FILE* escrever;
	

// =================================== Funções de Conveniência ===================================
point* calloc_arrayOfPoints(int size);
itemEa* calloc_edgeArray(int size);
int readFile(const char *nome, point *lista,  float eps);
int compare(const void *a, const void *b, int index);
float max(const float a, const float b);
void temposExecucao(double *utime, double *stime, double *total_time);
double currentTime();
int ordena (const void *a, const void *b);


// ===================================== Operações do Grafo =====================================
float euclideanDistance(point P, point P_l);
void montaGrafoOrdenaCPU(point *L, int *Va_i,int *Va_n, itemEa *Ea, float eps, int c);


// ================================= Heap struct and operations =================================
typedef struct element {
    point *p;
    //float priority;
}element;

typedef struct PriorityQueue {
    element *pq;      // pq stands for PriorityQueue
    int size;
}PriorityQueue;

PriorityQueue* createHeap(int size);
void destroyHeap(PriorityQueue *heap);
int heapSize(PriorityQueue *heap);
int heapIsFull(PriorityQueue *heap);
int heapIsEmpty(PriorityQueue *heap);
int insertHeap(PriorityQueue *heap, point *p);
void promoteElement(PriorityQueue *heap, int son);
point* getNextHeap(PriorityQueue *heap);
void demoteElement(PriorityQueue *heap, int dad);






void teste(float *X, int *Va_n, float eps, int how_large, int numDim, int idx);
#endif
