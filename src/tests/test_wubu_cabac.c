#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_cabac.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== CABAC Tests ===\n\n");

    printf("  g1_encode_skew...");
    {
        /* encode 100 mostly-zero bins → output should be very small */
        uint8_t* buf=calloc(1024,1);
        WuBuCabacEnc enc;
        wubu_cabac_init_encoder(&enc,buf,1024);
        
        CabacContext ctx={0,0}; /* state=0 (pLPS=0.5), MPS=0 */
        /* adapt: encode many zeros so MPS converges to 0 with low pLPS */
        for(int i=0;i<100;i++)
            wubu_cabac_encode_bin(&enc,&ctx,0);
        wubu_cabac_encode_terminate(&enc,1);
        
        long size=(long)enc.pos;
        printf("[%ld bytes for 100 skewed bins] ",size);
        CHECK(size>0&&size<20);  /* should be very compact */
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("  g2_bypass_uniform...");
    {
        uint8_t* buf=calloc(1024,1);
        WuBuCabacEnc enc;
        wubu_cabac_init_encoder(&enc,buf,1024);
        
        /* encode random bits via bypass — should use ~1 bit each */
        srand(42);
        for(int i=0;i<100;i++)
            wubu_cabac_encode_bypass(&enc,rand()%2);
        wubu_cabac_encode_terminate(&enc,1);
        
        long size=(long)enc.pos;
        printf("[%ld bytes for 100 uniform bins = %.1f bits/bin] ",size,(float)size*8/100);
        CHECK(size>=10&&size<=20); /* ~1.2 bits per bin including overhead */
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
