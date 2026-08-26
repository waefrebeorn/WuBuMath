/*
 * wubu_transform.c -- GROUP 7: Variable-size transforms
 *
 * G7.01: 4×4 integer DCT (H.264 core)
 * G7.02: 16×16 integer DCT
 * G7.03: 32×32 integer DCT (HEVC/VVC)
 * G7.05: Transform skip mode (identity)
 * G7.09: Quantization matrices per size/type
 */
#define M_PI 3.14159265358979f
#include "wubu_transform.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Generic N×N float DCT via matrix multiplication ===== */

/* precompute DCT basis matrix for given size */
static void dct_basis(int n,double* mat){
    /* C(u) = c(u) * cos((2x+1)*u*pi/(2n)) where c(0)=sqrt(1/n), c(u)=sqrt(2/n) */
    for(int u=0;u<n;u++){
        double cu=(u==0)?sqrt(1.0/n):sqrt(2.0/n);
        for(int x=0;x<n;x++)
            mat[u*n+x]=cu*cos((2*x+1)*u*M_PI/(2*n));
    }
}

/* forward transform: coeff = basis × input × basis^T (separable) */
static void dct_forward_n(const double* basis,const int16_t* input,
                           int16_t* output,int n){
    double tmp[n*n];
    
    /* rows: tmp = input × B^T (i.e., each row of input transformed) */
    for(int y=0;y<n;y++)
        for(int u=0;u<n;u++){
            double sum=0;
            for(int x=0;x<n;x++)
                sum+=input[y*n+x]*basis[u*n+x];
            tmp[y*n+u]=sum;
        }
    
    /* columns: output = B × tmp */
    for(int u=0;u<n;u++)
        for(int v=0;v<n;v++){
            double sum=0;
            for(int y=0;y<n;y++)
                sum+=basis[u*n+y]*tmp[y*n+v];
            output[v*n+u]=(int16_t)lround(sum);
        }
}

/* inverse transform: output = B^T × coeff × B */
static void dct_inverse_n(const double* basis,const int16_t* coeff,
                            int16_t* output,int n){
    double tmp[n*n];
    
    /* tmp = B^T × coeff */
    for(int y=0;y<n;y++)
        for(int x=0;x<n;x++){
            double sum=0;
            for(int u=0;u<n;u++)
                sum+=basis[u*n+y]*coeff[x*n+u];
            tmp[y*n+x]=sum;
        }
    
    /* output = tmp × B */
    for(int y=0;y<n;y++)
        for(int x=0;x<n;x++){
            double sum=0;
            for(int v=0;v<n;v++)
                sum+=tmp[y*n+v]*basis[v*n+x];
            output[y*n+x]=(int16_t)lround(sum);
        }
}

/* ===== Public API ===== */

void wubu_tr_forward(const int16_t* input,int16_t* output,int size){
    double basis[size*size];
    dct_basis(size,basis);
    dct_forward_n(basis,input,output,size);
}

void wubu_tr_inverse(const int16_t* coeff,int16_t* output,int size){
    double basis[size*size];
    dct_basis(size,basis);
    dct_inverse_n(basis,coeff,output,size);
}

/* transform skip: just copy */
void wubu_tr_skip(const int16_t* input,int16_t* output,int size){
    memcpy(output,input,sizeof(int16_t)*(size_t)(size*size));
}

/* quantize with scaling matrix (per-coefficient weights) */
void wubu_tr_quantize_m(const int16_t* coeffs,const uint8_t* qmat,
                          int16_t* output,int size,int qp){
    int shift=(qp/6)+4;
    int offset=1<<(shift-1);
    for(long i=0;i<(long)size*size;i++){
        int scaled=coeffs[i]*(int)qmat[i];
        output[i]=(int16_t)((scaled+offset)>>shift);
    }
}

void wubu_tr_dequantize_m(const int16_t* quantized,const uint8_t* qmat,
                            int16_t* output,int size,int qp){
    int shift=qp/6+4;
    for(long i=0;i<(long)size*size;i++)
        output[i]=(int16_t)(quantized[i]*qmat[i]<<shift>>4);
}

/* default flat quantization matrix */
static const uint8_t qmat_flat[64]={
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16
};

/* JPEG-style quality-aware quantization matrix for 8x8 */
static const uint8_t qmat_8x8[64]={
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99
};

const uint8_t* wubu_tr_get_qmat(int size,int intra){
    if(size==8)return qmat_8x8;
    return qmat_flat;
}
