/* wubu_topo_leakage_filter.h */
#ifndef WUBU_TOPO_LEAKAGE_FILTER_H
#define WUBU_TOPO_LEAKAGE_FILTER_H
#include <stddef.h>
/* Filter braid words: keep those with leakage < threshold. Returns count kept. */
int wubu_topo_filter(const double *leakages, int n, double threshold, int *kept);
#endif