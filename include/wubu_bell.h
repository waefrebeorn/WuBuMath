/* wubu_bell.h */
#ifndef WUBU_BELL_H
#define WUBU_BELL_H
#include <stddef.h>
/* Fill one of 4 Bell states into state (8 doubles complex) */
void wubu_bell_state(int idx, double *state);
#endif