#include "wubu_bell.h"
#include <string.h>
#include <math.h>

void wubu_bell_state(int idx, double *state) {
    for (int i=0;i<8;i++) state[i]=0;
    double s = 1.0/sqrt(2.0);
    switch (idx % 4) {
        case 0: state[0]=s; state[6]=s; break; /* (|00>+|11>)/sqrt2 */
        case 1: state[0]=s; state[6]=-s; break; /* (|00>-|11>)/sqrt2 */
        case 2: state[2]=s; state[4]=s; break; /* (|01>+|10>)/sqrt2 */
        case 3: state[2]=s; state[4]=-s; break; /* (|01>-|10>)/sqrt2 */
    }
}