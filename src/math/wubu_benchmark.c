#include "wubu_benchmark.h"
#include <time.h>
double wubu_bench(void (*fn)(void*), void *arg, int iters) {
    clock_t start = clock();
    for (int i = 0; i < iters; i++) fn(arg);
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}