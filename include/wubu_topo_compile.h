/* wubu_topo_compile.h */
#ifndef WUBU_TOPO_COMPILE_H
#define WUBU_TOPO_COMPILE_H
#include <stddef.h>
/* Topological compilation: given target 2-qubit gate (complex 4x4, 32 doubles),
   and a braid word length, returns a heuristic leakage estimate. */
double wubu_topo_leakage(const double *target, const double *braid_result);
#endif