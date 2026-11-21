#include "wubu_fusion_tree.h"

long long wubu_fib_fusion_paths(int n) {
    /* Fibonacci: F_0=1, F_1=1, F_n=F_{n-1}+F_{n-2} */
    if (n <= 1) return 1;
    long long a=1,b=1;
    for (int i=2;i<=n;i++){ long long c=a+b; a=b; b=c; }
    return b;
}