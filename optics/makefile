CC = gcc
GPUCC = nvcc
CFLAGS = -lm -Xcompiler -fopenmp -g
OBJDIR = obj

objects = $(addprefix $(OBJDIR)/, main.o optics.o optics_cuda.o grafo.o)
main = $(addprefix $(OBJDIR)/, main.o)
optics = $(addprefix $(OBJDIR)/, optics.o)
grafo = $(addprefix $(OBJDIR)/, grafo.o)
optics_cuda = $(addprefix $(OBJDIR)/, optics_cuda.o)

TARGET = goptics extract

all: $(TARGET)

extract: src/extractCluster.c
	$(CC) -Wall src/extractCluster.c -o $@

goptics: $(objects)
	$(GPUCC) $(CFLAGS) $(objects) -o $@

$(main): src/main.c include/optics.h include/optics_cuda.h
	$(GPUCC) $(CFLAGS) -c src/main.c -o $@

$(optics): src/optics.c include/optics.h include/grafo.h
	$(GPUCC) $(CFLAGS) -c src/optics.c -o $@ 

$(optics_cuda): src/optics_cuda.cu include/optics_cuda.h include/grafo.h
	$(GPUCC) $(CFLAGS) -c src/optics_cuda.cu -o $@

$(grafo): src/grafo.c include/grafo.h
	$(GPUCC) $(CFLAGS) -c src/grafo.c -o $@



clean:
	rm -f $(OBJDIR)/*.o clusters/*.txt goptics extract *~
