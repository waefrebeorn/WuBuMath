#include "wubu_anyon_fusion.h"

int wubu_fibonacci_fuse(int a, int b, int *outcomes) {
    /* vacuum rules */
    if (a == 0) { outcomes[0] = b; return 1; }
    if (b == 0) { outcomes[0] = a; return 1; }
    /* tau x tau -> vacuum or tau */
    outcomes[0] = 0; outcomes[1] = 1; return 2;
}