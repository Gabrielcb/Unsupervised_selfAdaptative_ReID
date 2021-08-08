#ifndef COMPONENTES_H
#define COMPONENTES_H

#ifdef __cplusplus
	extern "C" {
#endif

int GERA_NOVA;


itemEa* montaGrafoOrdenaGPU(point *L, int *Va_i, int *Va_n, float eps, int c);
itemEa* montaGrafoOrdenaCPUParalela(point *L, int *Va_i, int *Va_n, float eps, int c);
#ifdef __cplusplus
}
#endif

#endif
