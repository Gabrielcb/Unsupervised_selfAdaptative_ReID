#ifndef DBSCAN_H
#define DBSCAN_H

#include "grafo.h"

// =================================== Operações Optics ===================================
void expandClusterOrder(point *points, point *current, int minPts, float radius, itemEa *Ea, int *Va_i, int *Va_n, PriorityQueue *heap);
void setCoreDist(point *current, int minPts, itemEa *Ea, int *Va_i, int *Va_n);
void orderSeedsUpdate(point *points, point *o, float radius, itemEa *Ea, int *Va_i, int *Va_n, PriorityQueue *heap);

#endif
