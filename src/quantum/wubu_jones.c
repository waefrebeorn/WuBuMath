#include "wubu_jones.h"
#include "wubu_kauffman.h"
#include <math.h>

double wubu_jones_eval(const int *braid_word, int n, double t) {
    /* Jones polynomial via Kauffman bracket:
       V_L(t) = (-A)^{-3w} <K>_A  with A = t^{-1/4}, w = writhe.
       braid_word: array of crossing signs (+1/-1), n = number of crossings. */
    if (n == 0) return 1.0;
    /* Compute writhe */
    int w = 0;
    for (int i = 0; i < n; i++) w += braid_word[i];
    /* A = t^{-1/4} */
    double A = pow(t, -0.25);
    /* Kauffman bracket */
    double bracket = wubu_kauffman_bracket(braid_word, n, A);
    /* Jones = (-A)^{-3w} * bracket */
    double factor = pow(-A, -3.0 * w);
    return factor * bracket;
}
