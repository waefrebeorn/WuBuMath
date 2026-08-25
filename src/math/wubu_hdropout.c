/*
 * wubu_hdropout.c -- GAP-C045: Hyperbolic dropout
 * (tangent-space noise injection with on-manifold rescaling)
 *
 * Standard dropout zeroes coordinates and scales by 1/(1-p) — but on the
 * ball, zeroing coordinates moves points OFF the manifold. The correct
 * hyperbolic form:
 *   1. log_0: point -> tangent vector at origin (unbounded, Euclidean)
 *   2. apply standard dropout in tangent space (with inverted scaling)
 *   3. exp_0: back onto the ball
 *
 * This is exactly GGBall's "operate in tangent space" doctrine applied
 * to regularization — noise lives where noise is well-defined.
 */
#include "wubu_hdropout.h"
#include <stdlib.h>
#include <math.h>

int wubu_hd_apply(const float* x,int D,float c,float p_rate,
                   unsigned* seed,float* out){
    if(p_rate<0||p_rate>=1)return -1;
    float keep=1-p_rate;

    /* log_0(x): v = zn*x with zn = atanh(sqrt(c)|x|)/(sqrt(c)|x|) */
    float n2=0;
    for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float nx=sqrtf(n2);
    float v[512];
    int dd=D<512?D:512;
    if(nx>1e-10f){
        float arg=sqrtf(c)*nx;
        if(arg>0.999999f)arg=0.999999f;
        float zn=atanhf(arg)/(sqrtf(c)*nx);   /* standard log_0 */
        for(int d=0;d<dd&&d<512;d++)v[d]=zn*x[d];
    }else{
        for(int d=0;d<dd&&d<512;d++)v[d]=0;
    }

    /* tangent dropout + inverted scaling */
    for(int d=0;d<dd&&d<512;d++){
        *seed=*seed*1103515245u+12345u;
        float u=((float)((*seed>>16)%10000))/10000.0f;
        if(u<p_rate)v[d]=0;
        else v[d]/=keep;
    }
    for(int d=dd;d<D;d++)v[d]=0;

    /* exp_0(v): out = tanh(sqrt(c)|v|)/sqrt(c)|v| * v */
    float vn2=0;
    for(int d=0;d<D&&d<512;d++)vn2+=v[d]*v[d];
    float nv=sqrtf(vn2);
    if(nv>1e-10f){
        /* exp_0(v): |out| = tanh(sqrt(c)|v|)/sqrt(c), direction = v */
        float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
        for(int d=0;d<D&&d<512;d++)out[d]=coeff*v[d];
    }else{
        for(int d=0;d<D&&d<512;d++)out[d]=0;
    }
    for(int d=512;d<D;d++)out[d]=0;

    /* fp32 boundary cap */
    float n2c=0;for(int d=0;d<D;d++)n2c+=out[d]*out[d];
    if(n2c>0.99998f){
        float s=sqrtf(0.99998f/n2c);
        for(int d=0;d<D;d++)out[d]*=s;
    }
    return 0;
}
