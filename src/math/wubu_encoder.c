/* wubu_encoder.c -- C15 keystone: unified RDO encoder loop

Wires together all gated C15 components into a single-pass encoder:

  For each block (8×8):
  1. Predict  — inter (motion-compensated) or intra (DC/Vert/Horiz/Plane)
  2. Residual — original - prediction
  3. Transform — DCT forward
  4. Quantize  — with quality matrix
  5. Estimate bits — for RD cost
  6. Reconstruct — inverse DCT + add to prediction
  7. Accumulate bits

  Output: reconstructed frame + total bits used.
*/
#include "wubu_encoder.h"
#include "wubu_rdo.h"
#include "wubu_intra.h"
#include "wubu_transform.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BS 8

static int est_block_bits(const int16_t* coeffs){
    return wubu_rd_estimate_bits(coeffs,BS*BS);
}

int wubu_encode_frame(const uint8_t* orig,
                       const uint8_t* ref_past,
                       const uint8_t* ref_future,
                       int W,int H,int qp,
                       wubu_frame_type_t frame_type,
                       const int16_t* mv_grid,
                       uint8_t* recon,
                       long* total_bits){
    if(!orig||!recon||((frame_type!=WUBU_I_FRAME)&&!ref_past))return -1;
    *total_bits=0;
    int nbx=(W+BS-1)/BS, nby=(H+BS-1)/BS;
    int bi=0;

    for(int by=0;by<nby;by++){
        for(int bx=0;bx<nbx;bx++){
            int x0=bx*BS, y0=by*BS;
            int x1=x0+BS, y1=y0+BS;
            if(x1>W)x1=W; if(y1>H)y1=H;
            int pw=x1-x0, ph=y1-y0;
            int npw=pw*ph;

            /* prediction buffer */
            uint8_t pred[BS*BS]={0};

            if(frame_type==WUBU_I_FRAME){
                /* ---- intra prediction ---- */
                /* extract actual block for mode decision */
                uint8_t actual[BS*BS];
                for(int i=0;i<npw;i++){
                    int ry=y0+i/pw, rx=x0+i%pw;
                    if(ry<H&&rx<W)actual[i]=orig[ry*W+rx];
                }
                /* best intra mode by SAD */
                int mode=wubu_ip_best_mode(orig,actual,W,H,x0/BS,y0/BS);
                uint8_t blk_pred[BS*BS];
                wubu_ip_predict((uint8_t*)orig,W,H,x0/BS,y0/BS,mode,blk_pred);
                memcpy(pred,blk_pred,npw);
            }else{
                /* ---- inter prediction ---- */
                if(mv_grid){
                    int mvx=mv_grid[bi*2], mvy=mv_grid[bi*2+1];
                    const uint8_t* ref=(frame_type==WUBU_B_FRAME&&ref_future)?
                                        ref_future:ref_past;
                    if(ref){
                        for(int r=0;r<ph;r++)
                            for(int c=0;c<pw;c++){
                                int sx=x0+c-mvx, sy=y0+r-mvy;
                                if(sx<0)sx=0; if(sx>=W)sx=W-1;
                                if(sy<0)sy=0; if(sy>=H)sy=H-1;
                                pred[r*pw+c]=ref[sy*W+sx];
                            }
                    }
                }else if(ref_past){
                    for(int r=0;r<ph;r++)
                        for(int c=0;c<pw;c++){
                            int sx=x0+c, sy=y0+r;
                            if(sx<W&&sy<H)pred[r*pw+c]=ref_past[sy*W+sx];
                        }
                }
            }

            /* residual */
            int16_t res[BS*BS]={0};
            for(int i=0;i<npw;i++){
                int ry=y0+i/pw, rx=x0+i%pw;
                if(ry<H&&rx<W)
                    res[i]=(int16_t)((int)orig[ry*W+rx]-(int)pred[i]);
            }

            /* transform + quantize */
            int16_t coeffs[BS*BS];
            wubu_tr_forward(res,coeffs,BS);
            const uint8_t* qmat=wubu_tr_get_qmat(BS,frame_type==WUBU_I_FRAME);
            int16_t quant[BS*BS];
            wubu_tr_quantize_m(coeffs,qmat,quant,BS,qp);

            /* bits estimate */
            *total_bits+=est_block_bits(quant);

            /* reconstruct: dequantize + inverse DCT + add pred */
            int16_t dq[BS*BS], recon_blk[BS*BS];
            wubu_tr_dequantize_m(quant,qmat,dq,BS,qp);
            wubu_tr_inverse(dq,recon_blk,BS);
            for(int i=0;i<npw;i++){
                int ry=y0+i/pw, rx=x0+i%pw;
                if(ry<H&&rx<W){
                    int val=recon_blk[i]+pred[i];
                    if(val<0)val=0; if(val>255)val=255;
                    recon[ry*W+rx]=(uint8_t)val;
                }
            }
            bi++;
        }
    }
    return 0;
}

int wubu_rd_curve(const uint8_t* orig,
                   const uint8_t* ref_past,
                   const uint8_t* ref_future,
                   int W,int H,
                   const int* qp_vals,int n_qp,
                   wubu_frame_type_t frame_type,
                   const int16_t* mv_grid,
                   RdCurvePoint* curve){
    if(!orig||!curve)return -1;
    for(int i=0;i<n_qp;i++){
        uint8_t* recon=calloc((size_t)W*H,1);
        if(!recon)return -1;
        long bits=0;
        if(wubu_encode_frame(orig,ref_past,ref_future,W,H,
                             qp_vals[i],frame_type,mv_grid,
                             recon,&bits)!=0){free(recon);return -1;}
        long mse=0;
        for(long p=0;p<(long)W*H;p++){
            int d=orig[p]-recon[p]; mse+=d*d;
        }
        mse/=(double)W*H;
        double psnr=mse>0?10.0*log10(255.0*255.0/mse):99.9;
        curve[i].qp=qp_vals[i];
        curve[i].bits=bits;
        curve[i].psnr=psnr;
        free(recon);
    }
    return 0;
}
