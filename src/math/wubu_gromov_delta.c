#include "wubu_gromov_delta.h"
#include <math.h>
double wubu_gromov_delta(const double *D, int n) {
    double delta = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                double a = D[i*n+j], b = D[i*n+k], c = D[j*n+k];
                /* sort a,b,c */
                double x=a,y=b,z=c;
                if (x<y){double t=x;x=y;y=t;}
                if (y<z){double t=y;y=z;z=t;}
                if (x<y){double t=x;x=y;y=t;}
                double d = (x+y-z)/2.0;
                if (d > delta) delta = d;
            }
    return delta;
}