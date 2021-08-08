#include <stdio.h>
#include <stdlib.h>

typedef struct {
	float *X;

	float coreDist;
	float reachDist;

	int membership;
}orderedPts;

int main(int argc, char **argv){
	if(argc != 3){
		printf("ERROR!\n");
		printf("Argumentos inválidos, para executar entre com:\n");
		printf("./extract    ORDERED_POINTS_FILE_NAME	EPSILON'_VALUE\n");
		exit(EXIT_FAILURE);
	}
	
	int i = 0, j = 0;
	int clusterID = -1;

	int size;
	int numDim;
	float epsilon;
	float eLinha = atof(argv[2]);
	FILE* read = fopen(argv[1], "r");
	fscanf(read, "%d %f %d", &size, &epsilon, &numDim);


	if(eLinha > epsilon){
		printf("O epsilon' fornecido ( %f ) é maior que o epsilon ( %f ) com qual os pontos foram gerados.\n", eLinha, epsilon);
		printf("Forneça um epsilon' <= epsilon .\n");
		exit(EXIT_FAILURE);
	}


	orderedPts *OP = calloc(size, sizeof(orderedPts));
	for(i = 0; i < size; ++i){
		OP[i].X = (float*) calloc(numDim, sizeof(float));
	}

	// Setting i to zero to re-use the variable in the WHILE loop below
	i = 0;

	// Reading from file the data
	do{
		for(j = 0; j < numDim; ++j){			
			fscanf(read, "%f", &OP[i].X[j]);
		}

		fscanf(read,"%f %f", &OP[i].coreDist, &OP[i].reachDist);
		OP[i].membership = -1;
		i++;
	}while(!feof(read));

	fclose(read);

	// Extract cluster function
	for(i = 0; i < size; ++i) {
		if(OP[i].reachDist > eLinha) {
			if(OP[i].coreDist <= eLinha) {
				clusterID++;
				OP[i].membership = clusterID;
			}
			else {
				OP[i].membership = -1;
			}
		} else {
			OP[i].membership = clusterID;
		}
	}

	system("cd clusters && rm *.txt");

	
	// Write the clusters in the files
	char *nameFile = calloc(25, sizeof(char));
	FILE* write;
	for(i = 0; i < size; ++i) {
		if(OP[i].membership != -1) {
			sprintf(nameFile, "clusters/cluster%d.txt", OP[i].membership);
			write = fopen(nameFile, "a");
			for(j = 0; j < numDim; ++j)			
				fprintf(write, "%f ", OP[i].X[j]);
				
			fprintf(write, "\n");
			fclose(write);
		}
	}

	return 0;
}