#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jxl/encode.h>
#include <jxl/codestream_header.h>
static int W=176,H=144,NF=120;
static long read_ppm(const char*p,unsigned char*b){
    FILE*f=fopen(p,"rb");if(!f)return 0;
    char l[256];fgets(l,256,f);
    while(fgets(l,256,f)){if(l[0]!='#'){int w,h;sscanf(l,"%d %d",&w,&h);break;}}
    fgets(l,256,f);return (long)fread(b,1,(size_t)W*H*3,f);
}
static float bilin(const unsigned char*i,int w,int h,float x,float y,int c){
    int x0=(int)x,y0=(int)y;
    if(x<0)x=0;if(y<0)y=0;if(x0>=w-2)x0=w-2;if(y0>=h-2)y0=h-2;
    if(x0<0)x0=0;if(y0<0)y0=0;
    float fx=x-x0,fy=y-y0;const unsigned char*b=i+((size_t)y0*w+x0)*3+c;
    return b[0]*(1-fx)*(1-fy)+b[3]*fx*(1-fy)+b[w*3]*(1-fx)*fy+b[w*3+3]*fx*fy;
}
int main(void){
    unsigned char* frames=malloc((size_t)NF*W*H*3);
    for(int f=0;f<NF;f++){
        char p[256];snprintf(p,256,"wubq_frames/%04d.ppm",f+1);
        read_ppm(p,frames+(size_t)f*W*H*3);
    }
    
    /* also decode the JXL to measure PSNR */
    /* for now: just measure size at different distances */
    
    printf("=== ANIMATED JPEG XL QUALITY SWEEP ===\n");
    printf("%-10s %10s %8s\n","Distance","Bytes","Ratio");
    printf("───────── ────────── ────────\n");
    
    for(float dist=0.0f;dist<=10.0f;dist+=2.0f){
        if(dist==0.0f)dist=0.3f; /* near-lossless */
        
        JxlEncoder* enc=JxlEncoderCreate(NULL);
        JxlBasicInfo info;
        JxlEncoderInitBasicInfo(&info);
        info.xsize=W;info.ysize=H;
        info.num_color_channels=3;
        info.bits_per_sample=8;
        info.have_animation=JXL_TRUE;
        info.animation.tps_numerator=30;
        info.animation.tps_denominator=1;
        info.animation.num_loops=0;
        JxlEncoderSetBasicInfo(enc,&info);
        
        JxlColorEncoding color;
        JxlColorEncodingSetToSRGB(&color,JXL_FALSE);
        JxlEncoderSetColorEncoding(enc,&color);
        
        JxlEncoderFrameSettings* settings=JxlEncoderFrameSettingsCreate(enc,NULL);
        JxlEncoderSetFrameDistance(settings,dist);
        JxlEncoderFrameSettingsSetOption(settings,JXL_ENC_FRAME_SETTING_EFFORT,7);
        
        for(int fi=0;fi<NF;fi++){
            JxlFrameHeader fh;
            JxlEncoderInitFrameHeader(&fh);
            fh.duration=1;
            fh.is_last=(fi==NF-1)?JXL_TRUE:JXL_FALSE;
            JxlEncoderSetFrameHeader(settings,&fh);
            
            const uint8_t* data=frames+(size_t)fi*W*H*3;
            JxlPixelFormat fmt={3,JXL_TYPE_UINT8,JXL_LITTLE_ENDIAN,0};
            JxlEncoderAddImageFrame(settings,&fmt,data,(size_t)W*H*3);
        }
        
        size_t alloc=4*1024*1024;
        uint8_t* output=malloc(alloc);
        uint8_t* next_out=output;
        size_t avail=alloc;
        JxlEncoderStatus st=JxlEncoderProcessOutput(enc,&next_out,&avail);
        while(st==JXL_ENC_NEED_MORE_OUTPUT){
            size_t used=alloc-avail;
            alloc*=2;output=realloc(output,alloc);
            next_out=output+used;avail=alloc-used;
            st=JxlEncoderProcessOutput(enc,&next_out,&avail);
        }
        
        if(st==JXL_ENC_SUCCESS){
            size_t sz=alloc-avail;
            printf("%-10.1f %10zu %7.1fx\n",dist,sz,(float)(NF*(long)W*H*3)/sz);
        }else{
            printf("%-10.1f %10s %8s\n",dist,"FAILED","-");
        }
        free(output);
        JxlEncoderDestroy(enc);
    }
    
    free(frames);
    return 0;
}
