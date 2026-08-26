#include "wubu_screenc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== IBC Search =====
 * Searches the already-reconstructed region for a matching block.
 * BV (block vector) = displacement from current position to source.
 * Source must be entirely within reconstructed pixels (above/left). */
long wubu_ibc_search(const uint8_t* recon,int W,int H,
                      int bx,int by,int bs,
                      int search_range,
                      int* out_bx,int* out_by){
    long best_sad=(long)1<<40;
    *out_bx=-1;*out_by=-1;

    /* scan all valid source positions in reconstructed area */
    for(int sy=0;sy<=by-bs;sy++){
        for(int sx=0;sx<W-bs;sx++){
            /* skip if overlapping with current block or to its right on same row */
            if(sy+bs>by && sx+bs>bx)continue;
            
            long sad=0;
            for(int r=0;r<bs;r+=2){ /* sample every other row for speed */
                for(int c=0;c<bs;c+=2){
                    sad+=abs(recon[(size_t)(by+r)*W+(bx+c)]-
                             recon[(size_t)(sy+r)*W+(sx+c)]);
                    if(sad>=best_sad)goto skip;
                }
            }
            best_sad=sad;*out_bx=sx;*out_by=sy;
            if(sad==0)return 0;
            skip:;
        }
    }
    return best_sad;
}

void wubu_ibc_predict(const uint8_t* recon,int W,int H,
                       int src_bx,int src_by,int dst_bx,int dst_by,int bs,
                       uint8_t* output){
    for(int r=0;r<bs;r++)
        for(int c=0;c<bs;c++){
            int sy=src_by+r, sx=src_bx+c;
            int dy=dst_by+r, dx=dst_bx+c;
            if(sy<H&&sx<W&&dy<H&&dx<W)
                output[(size_t)dy*W+dx]=recon[(size_t)sy*W+sx];
        }
}

/* ===== Palette Mode ===== */

int wubu_palette_extract(const uint8_t* img,int W,int H,
                          int bx,int by,int bs,
                          uint8_t* palette,int max_colors){
    int count=0;
    for(int r=0;r<bs;r++)
        for(int c=0;c<bs;c++){
            uint8_t val=img[(size_t)(by+r)*W+(bx+c)];
            int found=0;
            for(int p=0;p<count;p++)
                if(palette[p]==val){found=1;break;}
            if(!found){
                if(count>=max_colors)return -1;
                palette[count++]=val;
            }
        }
    return count;
}

int wubu_palette_encode(const uint8_t* img,int W,int H,
                          int bx,int by,int bs,
                          const uint8_t* palette,int n_colors,
                          uint8_t* indices){
    int lookup[256];
    memset(lookup,-1,sizeof(lookup));
    for(int i=0;i<n_colors;i++)lookup[palette[i]]=i;
    
    for(int r=0;r<bs;r++)
        for(int c=0;c<bs;c++){
            uint8_t val=img[(size_t)(by+r)*W+(bx+c)];
            int idx=lookup[val];
            if(idx<0)return -1;
            indices[r*bs+c]=(uint8_t)idx;
        }
    return 0;
}

void wubu_palette_decode(const uint8_t* indices,const uint8_t* palette,
                           int n_colors,uint8_t* output,int n_pixels){
    for(int i=0;i<n_pixels;i++)
        output[i]=indices[i]<n_colors?palette[indices[i]]:0;
}

int wubu_palette_suitable(const uint8_t* img,int W,int H,
                            int bx,int by,int bs,int max_colors){
    uint8_t palette[256];
    int count=wubu_palette_extract(img,W,H,bx,by,bs,palette,max_colors);
    return count>0&&count<=max_colors/4;
}
