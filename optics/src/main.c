#include "../include/optics.h"
#include "../include/optics_cuda.h"


int main(int argc, char *argv[]) {

	// changed by Gabriel Bertocco from 7 to 8 on if statment
	if(argc != 8){ 
		printf("Invalid arguments, excecution stopped. How it should be:\n");
		printf("./goptics\n\tEPSILON_VALUE\n\tMINPTS_VALUE\n\tDATA_SET_FILE_NAME\n\tHOW_LARGE\n\tNUM_DIMENSIONS\n\t[CPU = 0] [CUDA = 1] [CPU_N = 2].\n");
		exit(0);
	}
	int i, j;

	
	// ======================== Variáveis de tempo ========================
	double inicio, final, final2, inicio2, inicioT, finalT;
	inicio = currentTime();
	tmt = 0;
	tot = 0;
	// ====================================================================

	
	float epsilon;
	int minPts;
	int totalPts;	// Total de pontos existentes


	// ================== Parametros ==================
	epsilon = atof(argv[1]);
	minPts = atoi(argv[2]);
	how_large = atoi(argv[4]);
	numDim = atoi(argv[5]);
	cuda = atoi(argv[6]);
	// ================================================


	// ==================================== Declarando os vetores necessários =========================================	
	point *dataSet = NULL;	// Todos os pontos. Estrutura necessária para manter as informações dos pontos
	int *Va_i = NULL;			// Va_i[Pontos] para cada ponto, qual seu inicio na lista Ea_ids e Ea_dist 
	int *Va_n = NULL;			// Va_n[Pontos] para cada ponto, quantos vizinhos cada ponto tem
	itemEa *Ea = NULL;			// Vetor de estrutura usada para armazenar vizinhos e distancias. USADA NA VERSÃO GPU! --- stands for Edgearray
	numEdges = 0;				// Somatório de todas as listas de adjacencias
	// ================================================================================================================


	// ==================================== Heap ====================================
	PriorityQueue *heap;
	heap = createHeap(how_large);
	// ==============================================================================


	// ============================== Definindo qual tipo de execução será realizada ===============================
	switch(cuda){
		case 0:
			cpu1 = 1;
			cuda = 0;
			cpuN = 0;
		break;

		case 1:
			cpu1 = 0;
			cuda = 1;
			cpuN = 0;
		break;

		case 2:
			cpu1 = 0;
			cuda = 0;
			cpuN = 1;
		break;

		default:
			printf("ERROR! Invalid input for execution type.\n");
			printf("\n\t[CPU = 0] [CUDA = 1] [CPU_N = 2].\n");
			exit(EXIT_FAILURE);
	}
	// ==============================================================================================================


	// ========================== Alocações Padrões ==========================
	dataSet = calloc_arrayOfPoints(how_large);
	for(i = 0; i < how_large; ++i)
		dataSet[i].X = (float*) calloc(numDim, sizeof(float));

	Va_i = (int*) calloc(how_large, sizeof(int));	
	Va_n = (int*) calloc(how_large, sizeof(int));
	// =======================================================================

	i = 0;

	// ============================== Lê pontos ==============================
	totalPts = readFile(argv[3], dataSet, epsilon);
	if (totalPts != how_large) {
		printf("Incoerência encontrada na quantidade de pontos lidos no arquivo TXT e a fornecida como parâmetro.\n");
		printf("Quantidade total de pontos informada: %d.\n", how_large);
		printf("Quantidade total de pontos lida: %d.\n", totalPts);
		exit(EXIT_FAILURE);
	}	
	// =======================================================================


	/*
	// ============================================
	// Porque se deve ordenar lista com base no X ?
	qsort(lista, how_large, sizeof(point), compare);
	for(i = 0; i < how_large; ++i){
		lista[i].id = i;
	}
	// ============================================
	*/


	// ====================== Alocação do Grafo ==============================
	// ======================== CPU Sequencial ===============================
	if(cpu1){
		Ea = calloc_edgeArray(numEdges); 
		montaGrafoOrdenaCPU(dataSet, Va_i, Va_n, Ea, epsilon, minPts);
	}
	
	// ======================== CPU Paralelo =================================
	if(cpuN) 
		Ea = montaGrafoOrdenaCPUParalela(dataSet, Va_i, Va_n, epsilon, minPts);

	// =========================== GPU CUDA ==================================
	if(cuda){
		Ea = montaGrafoOrdenaGPU(dataSet, Va_i, Va_n, epsilon, minPts);
	}

	// =======================================================================

	double topi,topf;
	double topt = 0.0;
	topi = currentTime();


	for(i = 0; i < how_large; ++i){
		//printf("%d\n", Va_n[i]);
	}


	// changed by Gabriel Bertocco from "OrderedPoints.txt" to argv[7]
	escrever = fopen(argv[7], "w");
	fprintf(escrever, "%d %.10f %d\n", how_large, epsilon, numDim);
	for(i = 0; i < how_large; ++i){
		if(!dataSet[i].processed){
			expandClusterOrder(dataSet, &dataSet[i], minPts, epsilon, Ea, Va_i, Va_n, heap);
		}
	}
	fclose(escrever);
	
	topf = currentTime();	
	topt = topf - topi;
	final = currentTime();

	// ======================== Imprime os tempos ===========================
	if(cpu1) {
		printf("CPU OPTICS\t%lf\n", topt);
		printf("CPU TOTAL\t%lf\n", final-inicio);
	} 
	if(cuda) { 
		printf("GPU OPTICS\t%lf\n", topt);
		printf("GPU TOTAL\t%lf\n", final-inicio);
	}
	// =======================================================================

	printf("\n");

	return 0;
}
