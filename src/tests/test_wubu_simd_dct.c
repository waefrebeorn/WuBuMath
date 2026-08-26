#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "wubu_simd_dct.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== SIMD DCT Tests ===\n\n");

    printf("  g1_dc_concentration...");
    {
        int16_t block[64],output[64];
        for(int i=0;i<64;i++)block[i]=100; /* constant */
        
        wubu_sdct_avx2(block,output);
        
        /* DC should be dominant, AC should be small relative to DC */
        int dc=abs(output[0]);
        printf("[DC=%d] ",dc);
        CHECK(dc>700); /* DC = 8*8*100/2 ≈ 800 with our scaling */
    }
    printf("PASS\n");passed++;

    printf("  g2_batch_processing...");
    {
        /* process 16 blocks and verify they all produce output */
        int n_blocks=16;
        int16_t* blocks=calloc((size_t)n_blocks*64,sizeof(int16_t));
        int16_t* outputs=calloc((size_t)n_blocks*64,sizeof(int16_t));
        
        srand(42);
        for(long i=0;i<(long)n_blocks*64;i++)blocks[i]=(int16_t)(rand()%512-256);
        
        long processed=wubu_sdct_batch(blocks,outputs,n_blocks);
        CHECK(processed==n_blocks);
        
        /* at least some coefficients should be nonzero */
        int nonzero=0;
        for(long i=0;i<(long)n_blocks*64;i++)
            if(outputs[i]!=0)nonzero++;
        CHECK(nonzero>n_blocks*32); /* most coefficients should be nonzero */
        free(blocks);free(outputs);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
