#include "wubu_braid_apply.h"

void wubu_braid_apply(const double *gates, const int *word, int n, double *state) {
    double s[2];
    for (int k = 0; k < n; k++) {
        const double *g = &gates[word[k]*4];
        s[0] = g[0]*state[0] + g[1]*state[1];
        s[1] = g[2]*state[0] + g[3]*state[1];
        state[0]=s[0]; state[1]=s[1];
    }
}