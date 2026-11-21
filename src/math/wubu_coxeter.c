#include "wubu_coxeter.h"
#include <stdlib.h>
#include <string.h>

/* Coxeter length for symmetric group S_n:
   Given a permutation (as array of n ints), compute minimum number of
   adjacent transpositions (s_i) to sort it = inversion count. */
int wubu_coxeter_length(const int *word, int n) {
    /* If word represents a permutation, compute inversion count.
       If word is a sequence of simple reflections (1..n-1), apply them
       to identity and count inversions of result. */
    /* Interpret word as sequence of adjacent transposition indices */
    int *perm = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = 0; i < n; i++) {  /* word has n entries as reflection indices */
        /* word[i] should be in [0, n-2] for S_n; clamp */
        int s = word[i];
        if (s < 0) s = 0;
        if (s > n-2 && n > 1) s = n-2;
        if (n > 1) {
            int tmp = perm[s];
            perm[s] = perm[s+1];
            perm[s+1] = tmp;
        }
    }
    /* Count inversions */
    int len = 0;
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            if (perm[i] > perm[j]) len++;
    free(perm);
    return len;
}
