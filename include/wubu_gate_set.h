/* wubu_gate_set.h */
#ifndef WUBU_GATE_SET_H
#define WUBU_GATE_SET_H
#include <stddef.h>
/* Fill standard gates as complex 2x2 (8 doubles: re00,im00,re01,im01,re10,im10,re11,im11) */
void wubu_gate_H(double *g);
void wubu_gate_X(double *g);
void wubu_gate_Z(double *g);
void wubu_gate_S(double *g);
void wubu_gate_T(double *g);
#endif