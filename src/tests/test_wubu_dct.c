#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_idct8x8.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Integer DCT 8x8 Tests ===\n\n");

    printf("  g1_dc_only...");
    {
        /* all-same block: only DC coefficient should be non-zero */
        int input[64],output[64];
        for(int i=0;i<64;i++)input[i]=100;
        wubu_dct8x8_forward(input,output);
        /* DC should be ~800 (100*64/8) */
        CHECK(abs(output[0])>50);
        for(int i=1;i<16;i++){  /* check first row AC terms */
            if(abs(output[i])>abs(output[0])/4){
                printf("[AC[%d]=%d too large vs DC=%d] ",i,output[i],output[0]);
                CHECK(0);
            }
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_roundtrip_quality...");
    {
        /* random block → DCT → quantize(q=3) → dequantize → IDCT */
        int input[64],coeffs[64],quantized[64],deq[64],output[64];
        unsigned seed=42;
        for(int i=0;i<64;i++){
            seed=seed*1103515245u+12345u;
            input[i]=(int)((seed>>16)%256)-128;  /* -128..127 range */
        }
        wubu_dct8x8_forward(input,coeffs);
        wubu_dct8x8_quantize(coeffs,3,quantized);
        wubu_dct8x8_dequantize(quantized,3,deq);
        wubu_dct8x8_inverse(deq,output);

        /* measure error */
        double mse=0;
        for(int i=0;i<64;i++){
            double d=input[i]-output[i];
            mse+=d*d;
        }
        mse/=64;
        float psnr=mse>0?(float)(10*log10(255.0*255.0/mse)):99;
        printf("[PSNR=%.1f dB] ",psnr);
        CHECK(psnr>20.0f);   /* reasonable reconstruction at q=3 */
    }
    printf("PASS\n");passed++;

    printf("  g3_quantization_produces_zeros...");
    {
        /* smooth gradient block → DCT → high quality quantize → many zeros in high freq */
        int input[64],coeffs[64],quantized[64];
        for(int r=0;r<8;r++)
            for(int c=0;c<8;c++)
                input[r*8+c]=r*10+c*5-32;  /* smooth gradient */

        wubu_dct8x8_forward(input,coeffs);
        wubu_dct8x8_quantize(coeffs,7,quantized);  /* aggressive quantization */

        int zeros=0;
        for(int i=16;i<64;i++)if(quantized[i]==0)zeros++;  /* count zeros past DC+first few */
        printf("[%d/48 high-freq zeros] ",zeros);
        CHECK(zeros>24);  /* most high-frequency coefficients should be zero */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
