/*
 * wubu_qc_zlib.c -- GAP-C059: Zlib entropy stage for the quaternion
 * codec (replacing fixed 2-bit residual with variable-length coding)
 *
 * The 2-bit residual wastes bits on zero residuals. Real codecs pass the
 * quantized residual through an entropy coder so zeros cost ~1 bit and
 * large values cost proportionally more. We add a zlib deflate stage.
 *
 * Pipeline: SLERP predict → residual → 4-bit pack → zlib compress
 * This should push compression from 3.8x toward 10x+ on smooth content.
 */
#define M_PI 3.14159265358979f
#include "wubu_qc_zlib.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* minimal DEFLATE via system zlib — we link -lz */
#include <zlib.h>

/* compress a buffer, returns compressed size or 0 on failure */
long wubu_qz_compress(const uint8_t* src,long src_size,
                       uint8_t** out_buf){
    uLongf dest_size=compressBound((uLong)src_size);
    uint8_t* dest=malloc(dest_size);
    if(compress2(dest,&dest_size,src,(uLong)src_size,Z_BEST_SPEED)!=Z_OK){
        free(dest);
        return 0;
    }
    *out_buf=dest;
    return (long)dest_size;
}

long wubu_qz_decompress(const uint8_t* src,long src_size,
                          uint8_t* dest,long dest_size){
    uLongf dest_len=dest_size;
    if(uncompress(dest,&dest_len,src,(uLong)src_size)!=Z_OK)
        return 0;
    return (long)dest_len;
}

/* encode with SLERP prediction + 4-bit residual + zlib */
long wubu_qz_encode(const uint8_t* frames,int n_frames,
                     int W,int H,float angle_step,
                     FILE* out){
    long total=0;
    /* write header */
    uint16_t w=W,h=H,nf=n_frames;
    fwrite("WUBZ",4,1,out);
    fwrite(&w,2,1,out);fwrite(&h,2,1,out);fwrite(&nf,2,1,out);

    /* KEY frame: RGB565 */
    const uint8_t* f0=frames;
    uint8_t* key=malloc((size_t)W*H*2);  /* RGB565 = 2 bytes per pixel */
    for(int p=0;p<W*H;p++){
        long i=(long)p*3,j=(long)p*2;
        uint16_t packed=((frames[i]>>3)<<11)|((frames[i+1]>>2)<<5)|(frames[i+2]>>3);
        key[j]=packed>>8;
        key[j+1]=packed&0xFF;
    }
    long raw_key_size=(long)W*H*2;
    /* write raw size so decoder knows, then zlib compress */
    fwrite(&raw_key_size,4,1,out);
    uint8_t* key_comp;
    long key_comp_size=wubu_qz_compress(key,raw_key_size,&key_comp);
    fwrite(&key_comp_size,4,1,out);
    fwrite(key_comp,1,(size_t)key_comp_size,out);
    total+=8+key_comp_size;
    free(key);free(key_comp);

    /* INTER frames */
    uint8_t* predicted=malloc((size_t)W*H*3);
    uint8_t* residual_raw=malloc((size_t)W*H*3/2+16);
    for(int fi=1;fi<n_frames;fi++){
        const uint8_t* prev=frames+(size_t)(fi-1)*W*H*3;
        const uint8_t* curr=frames+(size_t)fi*W*H*3;

        /* rotate prev forward to get prediction */
        float ca=cosf(angle_step),sa=sinf(angle_step);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                float dx=x-W/2.0f,dy=y-H/2.0f;
                float rx=dx*ca-dy*sa+W/2.0f;
                float ry=dx*sa+dy*ca+H/2.0f;
                int px=(int)fmodf(rx+1000*W,W);
                int py=(int)fmodf(ry+1000*H,H);
                for(int c=0;c<3;c++)
                    predicted[((size_t)y*W+x)*3+c]=prev[((size_t)py*W+px)*3+c];
            }

        /* compute 4-bit quantized residual: 2 values per byte, W*H/2 bytes total
         * but we have 3 channels × W*H pixels = W*H*3 values / 2 per byte = W*H*3/2 bytes
         * Use 4 bits per value, packing pairs of channels */
        /* pack R and G deltas into 1 byte per pixel (4+4 bits) */
        for(int p=0;p<W*H;p++){
            long i=(long)p*3,j=(long)p;  /* 1 byte per pixel */
            int16_t dr=(((curr[i]-predicted[i])>>4)+8)&0x0F;
            int16_t dg=(((curr[i+1]-predicted[i+1])>>4)+8)&0x0F;
            residual_raw[j]=(unsigned char)((dr<<4)|dg);
        }
        /* B channel: separate pass, 2 values per byte */
        long b_off=(long)W*H;
        for(long j=0;j<(long)(W*H/2);j++){
            int16_t db0=(((curr[(j*2)*3+2]-predicted[(j*2)*3+2])>>4)+8)&0x0F;
            int16_t db1=(((curr[(j*2+1)*3+2]-predicted[(j*2+1)*3+2])>>4)+8)&0x0F;
            residual_raw[b_off+j]=(unsigned char)((db0<<4)|db1);
        }

        /* write raw size + zlib compressed */
        long res_raw=(long)W*H*3/2;  /* actual packed size */
        fwrite(&res_raw,4,1,out);
        uint8_t* res_comp;
        long res_comp_size=wubu_qz_compress(residual_raw,res_raw,&res_comp);
        fwrite(&res_comp_size,4,1,out);
        fwrite(res_comp,1,(size_t)res_comp_size,out);
        total+=8+res_comp_size;
        free(res_comp);
    }

    free(predicted);free(residual_raw);
    return total;
}

/* decode */
void wubu_qz_decode(FILE* in,uint8_t* frames_out,
                     int n_frames,int W,int H,float angle_step){
    char magic[4];
    fread(magic,4,1,in);
    uint16_t w,h,nf;
    fread(&w,2,1,in);fread(&h,2,1,in);fread(&nf,2,1,in);

    /* read sizes + compressed KEY */
    long raw_key_size,key_comp_size;
    if(fread(&raw_key_size,4,1,in)!=1)return;
    if(fread(&key_comp_size,4,1,in)!=1)return;
    uint8_t* key_comp=malloc((size_t)key_comp_size);
    if(fread(key_comp,1,(size_t)key_comp_size,in)!=(size_t)key_comp_size){free(key_comp);return;}
    uint8_t* key_raw=malloc((size_t)raw_key_size);
    uLongf dest_len=raw_key_size;
    if(uncompress(key_raw,&dest_len,key_comp,(uLong)key_comp_size)!=Z_OK){
        free(key_raw);free(key_comp);return;
    }
    free(key_comp);
    /* dequantize RGB565 */
    for(long j=0,i=0;j<(long)W*H*3;i+=2,j+=3){
        unsigned r5=(key_raw[i]>>3)&0x1F;
        unsigned g3=key_raw[i]&0x07;
        unsigned g5=((key_raw[i+1]>>6)&0x03)|(g3<<2);
        unsigned b5=key_raw[i+1]&0x1F;
        frames_out[j]=(uint8_t)(r5*255/31);
        frames_out[j+1]=(unsigned char)(g5*255/63);
        frames_out[j+2]=(unsigned char)(b5*255/31);
    }
    free(key_raw);

    /* INTER frames */
    uint8_t* predicted=malloc((size_t)W*H*3+16);
    uint8_t* residual_raw=malloc((size_t)W*H*2+16);
    for(int fi=1;fi<n_frames;fi++){
        uint8_t* prev=frames_out+(size_t)(fi-1)*W*H*3;
        uint8_t* curr=frames_out+(size_t)fi*W*H*3;

        /* rotate prev to reconstruct prediction */
        float ca=cosf(angle_step),sa=sinf(angle_step);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                float dx=x-W/2.0f,dy=y-H/2.0f;
                float rx=dx*ca-dy*sa+W/2.0f;
                float ry=dx*sa+dy*ca+H/2.0f;
                int px=(int)fmodf(rx+1000*W,W);
                int py=(int)fmodf(ry+1000*H,H);
                for(int c=0;c<3;c++)
                    predicted[((size_t)y*W+x)*3+c]=prev[((size_t)py*W+px)*3+c];
            }

        /* read sizes + compressed residual */
        long res_raw,res_comp_size;
        if(fread(&res_raw,4,1,in)!=1)return;
        if(fread(&res_comp_size,4,1,in)!=1)return;
        uint8_t* res_comp=malloc((size_t)res_comp_size);
        if(fread(res_comp,1,(size_t)res_comp_size,in)!=(size_t)res_comp_size){free(res_comp);return;}
        uLongf dest_len=res_raw;
        if(uncompress(residual_raw,&dest_len,res_comp,(uLong)res_comp_size)!=Z_OK){
            free(res_comp);continue;
        }
        free(res_comp);

        /* apply residual: R,G from first W*H bytes, B from second part */
        for(int p=0;p<W*H;p++){
            long i=(long)p*3,j=(long)p;
            int16_t dr=((residual_raw[j]>>4)&0x0F)-8;
            int16_t dg=(residual_raw[j]&0x0F)-8;
            int v0=predicted[i]+dr*16;
            int v1=predicted[i+1]+dg*16;
            curr[i]=(uint8_t)(v0<0?0:(v0>255?255:v0));
            curr[i+1]=(uint8_t)(v1<0?0:(v1>255?255:v1));
        }
        for(long j=0;j<(long)(W*H/2);j++){
            int p=j*2;
            int16_t db0=((residual_raw[(long)W*H+j]>>4)&0x0F)-8;
            int16_t db1=(residual_raw[(long)W*H+j]&0x0F)-8;
            int v0=predicted[p*3+2]+db0*16;
            int v1=predicted[(p+1)*3+2]+db1*16;
            curr[p*3+2]=(uint8_t)(v0<0?0:(v0>255?255:v0));
            curr[(p+1)*3+2]=(uint8_t)(v1<0?0:(v1>255?255:v1));
        }
    }
    free(predicted);free(residual_raw);
}
