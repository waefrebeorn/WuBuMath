#include "wubu_gate_set.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void zero(double*g){ for(int i=0;i<8;i++) g[i]=0; }

void wubu_gate_H(double *g){ zero(g); double s=1.0/sqrt(2.0); g[0]=s; g[2]=s; g[4+0]=s; g[4+2]=-s; }
void wubu_gate_X(double *g){ zero(g); g[2]=1; g[4+0]=1; }
void wubu_gate_Z(double *g){ zero(g); g[0]=1; g[4+2]=-1; }
void wubu_gate_S(double *g){ zero(g); g[0]=1; g[5]=1; }
void wubu_gate_T(double *g){ zero(g); g[0]=1; g[4+2]=cos(M_PI/4.0); g[5]=sin(M_PI/4.0); }
