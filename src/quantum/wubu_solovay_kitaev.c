#include "wubu_solovay_kitaev.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void qmul(const double*a,const double*b,double*o){
    o[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    o[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    o[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    o[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}
static double qdist(const double*a,const double*b){
    double d=0; for(int i=0;i<4;i++){double x=a[i]-b[i];d+=x*x;} return sqrt(d);
}

int wubu_sk_approx(const double *target, const double *gates, int ngates,
                   int max_len, int *out_word) {
    /* BFS over words up to max_len, track best */
    /* state: current quaternion (4), parent index, gate used */
    /* We do iterative deepening-ish: try all words length 1..max_len */
    double best_q[4]; for(int i=0;i<4;i++) best_q[i]=target[i];
    int best_len=0; double best_err=1e9;
    /* length 0 = identity */
    double id[4]={1,0,0,0};
    if (qdist(id,target)<best_err){ best_err=qdist(id,target); best_len=0; }
    /* DFS */
    /* recursive */
    /* use stack */
    /* simple recursion via helper */
    /* We'll do recursive depth-first */
    /* closure via static function pointer not needed; inline recursion with arrays */
    /* Implement iterative over lengths using BFS queue of states */
    /* state struct */
    typedef struct { double q[4]; int len; int word[8]; } st;
    /* max 8 length */
    if (max_len>8) max_len=8;
    /* BFS */
    /* queue */
    int cap=100000;
    st *Q=malloc(cap*sizeof(st));
    int head=0,tail=0;
    st s0; for(int i=0;i<4;i++) s0.q[i]=id[i]; s0.len=0;
    Q[tail++]=s0;
    while(head<tail){
        st cur=Q[head++];
        if (cur.len>=max_len) continue;
        for (int g=0;g<ngates;g++){
            st nx=cur;
            double gq[4]={gates[g*4],gates[g*4+1],gates[g*4+2],gates[g*4+3]};
            qmul(cur.q,gq,nx.q);
            nx.len=cur.len+1;
            nx.word[cur.len]=g;
            double e=qdist(nx.q,target);
            if (e<best_err){ best_err=e; best_len=nx.len; for(int i=0;i<nx.len;i++) out_word[i]=nx.word[i]; }
            if (tail<cap) Q[tail++]=nx;
        }
    }
    free(Q);
    return best_len;
}