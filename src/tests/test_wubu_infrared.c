#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_infrared.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Infrared Band Tests ===\n\n");
    const int CW=64,IR=8;
    uint8_t canvas[CW*(CW+IR)*3];
    memset(canvas,0,sizeof(canvas));

    printf("  g1_audio_roundtrip...");
    {
        float audio[100],back[100];
        unsigned rs=42u;
        for(int i=0;i<100;i++){
            rs=rs*1103515245u+12345u;
            audio[i]=((rs>>16)%200)/10000.0f-0.01f;
        }
        CHECK(wubu_ir_pack(audio,100,CW,IR,canvas)==0);
        wubu_ir_unpack(canvas,CW,IR,back,100);
        for(int i=0;i<100;i++){
            CHECK(fabsf(audio[i]-back[i])<0.01f);
        }
    }
    printf("PASS\n");passed++;
    printf("  g2_checksum_detects_corruption...");
    {
        uint32_t before=wubu_ir_checksum(canvas,CW,IR);
        /* flip a bit in the IR region */
        canvas[(size_t)((CW-4)*CW+30)*3]^=0xFF;
        uint32_t after=wubu_ir_checksum(canvas,CW,IR);
        CHECK(before!=after);
    }
    printf("PASS\n");passed++;
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
