/* wubu_benchmark.h */
#ifndef WUBU_BENCHMARK_H
#define WUBU_BENCHMARK_H
#include <stddef.h>
/* Benchmark: time a function pointer over N iterations, return seconds */
double wubu_bench(void (*fn)(void*), void *arg, int iters);
#endif