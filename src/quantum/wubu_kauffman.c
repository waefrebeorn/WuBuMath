#include "wubu_kauffman.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Kauffman bracket state sum:
   Given a link diagram as a list of crossings (each crossing has 4 arc endpoints),
   enumerate all 2^n states (A or B smoothing at each crossing),
   compute weight = A^{a} * A^{-b} * (-A^2 - A^{-2})^{loops-1}
   where a = #A smoothings, b = #B smoothings, loops = number of resulting loops.
   
   Simplified: for a braid word, convert to crossings and compute.
   Here we implement for a simple 2-crossing example and general n-crossing
   where crossings are given as an array of signs (+1 or -1). */

/* Count loops in a state given smoothing choices */
static int count_loops(const int *smoothings, int n_crossings) {
    /* Simplified: each smoothing either connects (a,b)-(c,d) or (a,c)-(b,d).
       For a single component link, loops = (n_crossings/2) + 1 - (number of B smoothings)/2
       This is a simplification; real implementation needs planar graph traversal. */
    int nA = 0, nB = 0;
    for (int i = 0; i < n_crossings; i++) {
        if (smoothings[i] == 0) nA++; else nB++;
    }
    /* For a 2-component Hopf link type: loops = n_crossings/2 + 1 */
    return n_crossings / 2 + 1 - nB / 2;
}

double wubu_kauffman_bracket(const int *diagram, int n, double A) {
    if (n == 0) return 1.0;
    /* diagram[i] = +1 (positive crossing) or -1 (negative crossing) */
    double Am1 = 1.0 / A;
    double d = -(A*A + Am1*Am1); /* loop value */
    double result = 0.0;
    int nstates = 1 << n;  /* 2^n states */
    for (int state = 0; state < nstates; state++) {
        int smoothings[64];
        int nA = 0, nB = 0;
        for (int i = 0; i < n && i < 64; i++) {
            smoothings[i] = (state >> i) & 1;
            if (smoothings[i] == 0) nA++; else nB++;
        }
        int loops = count_loops(smoothings, n);
        double weight = pow(A, nA) * pow(Am1, nB) * pow(d, loops - 1);
        result += weight;
    }
    return result;
}
