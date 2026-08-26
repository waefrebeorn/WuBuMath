#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_cabac.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== CABAC Tests ===\n\n");

    printf("  g1_skew_compression...");
    {
        /* 200 bins with 95% zeros → should be ~9 bytes (0.36 bits/bin!) */
        uint8_t* buf=calloc(4096,1);
        WuBuCabacEnc e;
        wubu_cabac_start(&e,buf,4096);
        
        CabacContext ctx={0,0};
        srand(42);
        for(int i=0;i<200;i++){
            int b=rand()%100<95?0:1;
            wubu_cabac_encode_bin(&e,&ctx,b);
        }
        long size=wubu_cabac_finish(&e);
        float bits_per_bin=(float)size*8/200;
        printf("[%ld bytes = %.2f bits/bin] ",size,bits_per_bin);
        CHECK(size>0&&size<30);
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("  g2_uniform_bypass...");
    {
        uint8_t* buf=calloc(4096,1);
        WuBuCabacEnc e;
        wubu_cabac_start(&e,buf,4096);
        
        srand(42);
        for(int i=0;i<100;i++)
            wubu_cabac_encode_bypass(&e,rand()%2);
        long size=wubu_cabac_finish(&e);
        float bits_bin=(float)size*8/100;
        printf("[%ld bytes = %.1f bits/bin] ",size,bits_bin);
        CHECK(bits_bin>=1.0f&&bits_bin<=2.0f); /* uniform → ~1 bit each */
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
