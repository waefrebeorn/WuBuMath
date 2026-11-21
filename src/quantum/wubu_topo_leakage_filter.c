#include "wubu_topo_leakage_filter.h"
int wubu_topo_filter(const double *leakages, int n, double threshold, int *kept) {
    int c = 0;
    for (int i = 0; i < n; i++) {
        if (leakages[i] < threshold) { kept[c++] = i; }
    }
    return c;
}