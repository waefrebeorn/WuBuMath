/*
 * train_wubu_codec.c -- Per-patch autoencoder with full backpropagation
 *
 * Architecture (per patch, shared weights across all 256 patches):
 *   patch[48] → Linear[48→32] → LayerNorm → [32]
 *   → 4 flow MLP levels (W1→GELU→W2→residual) → [32]
 *   → latent[32]
 *   → Decode: Linear[32→48] → reconstructed patch[48]
 *
 * Loss: MSE over all 256 patches
 * Optimizer: Adam
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum { IMG_H=64, IMG_W=64, PS=4, NPH=16, NPW=16, NP=256, PP=48,
       DIM=32, LDIM=32, LVL=4, FH=32,
       BATCH=16, EPOCHS=200, NSAMP=512 };

static unsigned rng_v = 77;
static float rf(void){ rng_v=rng_v*1103515245u+12345u; return(float)(rng_v&0x7FFF)/32768.0f; }
static float rn(void){ float u=rf(); if(u<1e-6f)u=1e-6f; return sqrtf(-2.0f*logf(u))*cosf(2.0f*M_PI*rf()); }

static float* data;

static void gen(float* img, int id) {
    int p=id%12; float ph=(float)(id/12)*0.03f;
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
        int i=(y*IMG_W+x)*3;
        float u=(float)x/(IMG_W-1), v=(float)y/(IMG_H-1), n=rn()*0.015f;
        switch(p){
            case 0:img[i]=u+n;img[i+1]=.5+n;img[i+2]=1-u+n;break;
            case 1:img[i]=.3+n;img[i+1]=v+n;img[i+2]=(1-v)*.7+n;break;
            case 2:img[i]=((x/8+y/8)%2)?.8:.2+n;img[i+1]=((x/8)%2)?.7:.3+n;img[i+2]=u*v+n;break;
            case 3:{float r=sqrtf((u-.5f)*(u-.5f)+(v-.5f)*(v-.5f))*2;
                    img[i]=fminf(1,r+n);img[i+1]=fminf(1,1-r+n);img[i+2]=.5+.5*sinf(r*6.28f)+n;break;}
            case 4:img[i]=.5+.5*sinf(u*12.56f+ph)+n;img[i+1]=.5+.5*cosf(v*12.56f+ph)+n;
                   img[i+2]=.5+.5*sinf((u+v)*6.28f)+n;break;
            case 5:{float d=(u+v)*.5;
                    img[i]=.5+.5*sinf(d*12.56f+ph)+n;img[i+1]=.5+.5*cosf(d*18.84f+ph)+n;
                    img[i+2]=.3+.2*sinf(d*6.28f)+n;break;}
            case 6:{int b=(x<4||x>=IMG_W-4||y<4||y>=IMG_H-4);
                    img[i]=b?.9:.2+n;img[i+1]=b?.1:.6+n;img[i+2]=b?.1:.3+n;break;}
            case 7:{float bx=u-.3,by=v-.7,g1=expf(-(bx*bx+by*by)*20);
                    bx=u-.7;by=v-.3;float g2=expf(-(bx*bx+by*by)*15);
                    img[i]=g1+n;img[i+1]=g2+n;img[i+2]=(g1+g2)*.5+n;break;}
            case 8:{float a=atan2f(v-.5,u-.5);
                    img[i]=.5+.5*sinf(a*3+ph)+n;img[i+1]=.5+.5*cosf(a*5+ph)+n;
                    img[i+2]=.5+.5*sinf(a*7+ph)+n;break;}
            case 9:{float c=((x/6)%2==(y/6)%2)?.7:.3;
                    img[i]=c*sinf(u*12.56f+ph)+.5+n;img[i+1]=c*cosf(v*12.56f+ph)+.5+n;
                    img[i+2]=c*sinf((u+v)*8.88f)+.5+n;break;}
            case 10:{float r=sqrtf((u-.5)*(u-.5)+(v-.5)*(v-.5))*20;
                    img[i]=.5+.5*sinf(r+ph)+n;img[i+1]=.5+.5*cosf(r*.7+ph)+n;
                    img[i+2]=.5+.5*sinf(r*1.3+ph*2)+n;break;}
            case 11:{int cx=(x>IMG_W/2-8&&x<IMG_W/2+8),cy=(y>IMG_H/2-8&&y<IMG_H/2+8);
                    img[i]=(cx||cy)?.8:.2+n;img[i+1]=(cx)?.1:.6+n;img[i+2]=(cy)?.1:.6+n;break;}
        }
        img[i]=fminf(1,fmaxf(0,img[i]));
        img[i+1]=fminf(1,fmaxf(0,img[i+1]));
        img[i+2]=fminf(1,fmaxf(0,img[i+2]));
    }
}

static float gelu_f(float x){return .5*x*(1+tanhf(.7978845608*x*(1+.044715*x*x)));}
static float gelu_b(float x){float c=.7978845608,inner=c*x*(1+.044715*x*x),t=tanhf(inner);return .5*(1+t)+.5*x*(1-t*t)*c*(1+3*.044715*x*x);}

typedef struct {
    float pw[PP*DIM], pb[DIM], plg[DIM], plb[DIM];
    float fw1[LVL][DIM*FH], fb1[LVL][FH];
    float fw2[LVL][FH*DIM], fb2[LVL][DIM];
    float dw[NP*PP*LDIM], db[NP*PP];
    float pwm[PP*DIM], pwv[PP*DIM], pbm[DIM], pbv[DIM], plgm[DIM], plgv[DIM], plbm[DIM], plbv[DIM];
    float fwm[LVL][DIM*FH], fwv[LVL][DIM*FH], fbm[LVL][FH], fbv[LVL][FH];
    float fwm2[LVL][FH*DIM], fwv2[LVL][FH*DIM], fbm2[LVL][DIM], fbv2[LVL][DIM];
    float dwm[NP*PP*LDIM], dwv[NP*PP*LDIM], dbm[NP*PP], dbv[NP*PP];
    int step;
} Model;

static Model model;

static void init_model(void){
    memset(&model,0,sizeof(model));
    for(int i=0;i<DIM;i++)model.plg[i]=1;
    float lim=sqrtf(6.0/(PP+DIM));
    for(int i=0;i<PP*DIM;i++)model.pw[i]=(rf()*2-1)*lim;
    for(int i=0;i<NP*PP*LDIM;i++)model.dw[i]=rn()*0.005f;
    for(int l=0;l<LVL;l++){
        for(int h=0;h<FH;h++)for(int d=0;d<DIM;d++)model.fw1[l][h*DIM+d]=rn()*0.01;
        /* fw2 zero init */
    }
    model.step=0;
}

/* Forward + backward for one sample.
 * Returns MSE loss, accumulates gradients. */

static float forward_backward(float* img,
                                float* gpw, float* gpb,
                                float* gplg, float* gplb,
                                float* gfw1, float* gfb1,
                                float* gfw2, float* gfb2,
                                float* gdw, float* gdb) {
    float loss=0;

    /* Per-patch forward + backward (fused) */
    for(int p=0; p<NP; p++) {
        /* Extract patch pixels */
        float pd[PP];
        int py=p/NPW, px=p%NPW, idx=0;
        for(int dy=0; dy<PS; dy++) for(int dx=0; dx<PS; dx++) {
            int iy=py*PS+dy, ix=px*PS+dx, ii=(iy*IMG_W+ix)*3;
            pd[idx++]=img[ii]; pd[idx++]=img[ii+1]; pd[idx++]=img[ii+2];
        }

        /* Linear: pre = W @ patch + b */
        float pre[DIM];
        for(int o=0; o<DIM; o++) {
            float s=model.pb[o];
            for(int i=0; i<PP; i++) s+=model.pw[o*PP+i]*pd[i];
            pre[o]=s;
        }

        /* LayerNorm */
        float mean=0; for(int i=0;i<DIM;i++)mean+=pre[i]; mean/=DIM;
        float var=0; for(int i=0;i<DIM;i++){float d=pre[i]-mean;var+=d*d;} var/=DIM;
        float inv=1/sqrtf(var+1e-5f);
        float pl[DIM];
        for(int i=0;i<DIM;i++) pl[i]=model.plg[i]*(pre[i]-mean)*inv+model.plb[i];

        /* 4 flow MLP levels */
        float v[DIM];
        memcpy(v, pl, DIM*sizeof(float));

        float v_in[LVL][DIM], hb_store[LVL][FH], hba_store[LVL][FH];

        for(int l=0; l<LVL; l++) {
            memcpy(v_in[l], v, DIM*sizeof(float));
            float hb[FH], hba[FH], fv[DIM];
            for(int h=0; h<FH; h++) {
                float s=model.fb1[l][h];
                for(int i=0; i<DIM; i++) s+=model.fw1[l][h*DIM+i]*v[i];
                hb[h]=s; hba[h]=gelu_f(s);
            }
            for(int o=0; o<DIM; o++) {
                float s=model.fb2[l][o];
                for(int h=0; h<FH; h++) s+=model.fw2[l][o*FH+h]*hba[h];
                fv[o]=s;
            }
            for(int i=0; i<DIM; i++) v[i]+=fv[i];
            memcpy(hb_store[l], hb, sizeof(hb));
            memcpy(hba_store[l], hba, sizeof(hba));
        }

        /* Latent = first LDIM dims of v */
        float lat[LDIM];
        for(int i=0; i<LDIM; i++) lat[i]=v[i];

        /* Decode + loss */
        float dlat[LDIM];
        memset(dlat, 0, sizeof(dlat));
        for(int o=0; o<PP; o++) {
            float pred=model.db[p*PP+o];
            for(int i=0; i<LDIM; i++) pred+=model.dw[p*PP*LDIM+o*LDIM+i]*lat[i];
            /* GT */
            int ch=o%3;
            int ox=o/3%PS, oy=o/3/PS;
            int iy=py*PS+oy, ix=px*PS+ox;
            float gt=img[(iy*IMG_W+ix)*3+ch];
            float d=pred-gt;
            loss+=d*d;
            /* Decoder backward */
            float grad=2.0f*d/(float)(NP*PP);
            gdb[p*PP+o]+=grad;
            for(int i=0; i<LDIM; i++) {
                gdw[p*PP*LDIM+o*LDIM+i]+=grad*lat[i];
                dlat[i]+=grad*model.dw[p*PP*LDIM+o*LDIM+i];
            }
        }

        /* === Encoder backward (this patch) === */
        float dv[DIM];
        for(int i=0; i<DIM && i<LDIM; i++) dv[i]=dlat[i];
        for(int i=LDIM; i<DIM; i++) dv[i]=0;

        for(int l=LVL-1; l>=0; l--) {
            /* Residual backward */
            float dfv[DIM];
            for(int i=0; i<DIM; i++) dfv[i]=dv[i];

            /* W2 backward */
            float dhba[FH];
            memset(dhba, 0, sizeof(dhba));
            for(int o=0; o<DIM; o++) {
                float g=dfv[o];
                gfb2[o]+=g;
                for(int h=0; h<FH; h++) {
                    gfw2[o*FH+h]+=g*hba_store[l][h];
                    dhba[h]+=g*model.fw2[l][o*FH+h];
                }
            }

            /* GELU backward */
            float dhb[FH];
            for(int h=0; h<FH; h++) dhb[h]=dhba[h]*gelu_b(hb_store[l][h]);

            /* W1 backward */
            for(int h=0; h<FH; h++) {
                float g=dhb[h];
                gfb1[h]+=g;
                for(int i=0; i<DIM; i++) {
                    gfw1[h*DIM+i]+=g*v_in[l][i];
                    dv[i]+=g*model.fw1[l][h*DIM+i];
                }
            }
        }

        /* LayerNorm backward */
        float dvar=0;
        for(int i=0; i<DIM; i++) {
            float dxhat=dv[i]*model.plg[i];
            dvar+=dxhat*(pre[i]-mean);
        }
        dvar*=-0.5f*powf(var,-1.5f);

        for(int i=0; i<DIM; i++) {
            float dxhat=dv[i]*model.plg[i];
            float dx=dxhat*inv;
            dx+=dxhat*(-0.5f)*inv*inv*inv*2.0f*(pre[i]-mean)*dvar;
            float dmean=0;
            for(int j=0;j<DIM;j++){
                float dyhat=dv[j]*model.plg[j];
                dmean+=-dyhat*inv;
                dmean+=-dyhat*inv*2.0f*(pre[j]-mean)*dvar;
            }
            dmean/=(float)DIM;
            dx+=dmean;

            gplg[i]+=(pre[i]-mean)*inv*dv[i];
            gplb[i]+=dv[i];
            gpb[i]+=dx;
            for(int j=0;j<PP;j++) gpw[i*PP+j]+=dx*pd[j];
        }
    }

    return loss / (float)(NP*PP);
}


int main(void){
    printf("WuBu Codec Training — Per-Patch Autoencoder\n");
    printf("============================================\n\n");
    init_model();

    printf("Generating %d training samples...\n",NSAMP);
    data=(float*)malloc((size_t)(NSAMP*IMG_H*IMG_W*3)*sizeof(float));
    for(int n=0;n<NSAMP;n++)gen(data+n*IMG_H*IMG_W*3,n);
    printf("Dataset ready: %.1f MB\n\n",(float)(NSAMP*IMG_H*IMG_W*3*4)/(1024*1024));

    float *gpw=calloc(PP*DIM,sizeof(float)),*gpb=calloc(DIM,sizeof(float));
    float *gplg=calloc(DIM,sizeof(float)),*gplb=calloc(DIM,sizeof(float));
    float *gfw1=calloc(DIM*FH,sizeof(float)),*gfb1=calloc(FH,sizeof(float));
    float *gfw2=calloc(FH*DIM,sizeof(float)),*gfb2=calloc(DIM,sizeof(float));
    float *gdw=calloc(NP*PP*LDIM,sizeof(float)),*gdb=calloc(NP*PP,sizeof(float));

    FILE* log=fopen("/home/wubu/WuBuMath/output/train_log.csv","w");
    fprintf(log,"epoch,loss,psnr\n");

    printf("Training: %d epochs, batch %d\n",EPOCHS,BATCH);
    printf("Image: %dx%d, patches: %d, latent: %d bytes\n",IMG_H,IMG_W,NP,LDIM);
    printf("Compression: %.0f:1\n\n",(float)(IMG_H*IMG_W*3*4)/(float)LDIM);

    for(int ep=0;ep<EPOCHS;ep++){
        float el=0;int nb=NSAMP/BATCH;
        for(int b=0;b<nb;b++){
            int s=b*BATCH;float bl=0;
            memset(gpw,0,sizeof(float)*PP*DIM);memset(gpb,0,sizeof(float)*DIM);
            memset(gplg,0,sizeof(float)*DIM);memset(gplb,0,sizeof(float)*DIM);
            memset(gfw1,0,sizeof(float)*DIM*FH);memset(gfb1,0,sizeof(float)*FH);
            memset(gfw2,0,sizeof(float)*FH*DIM);memset(gfb2,0,sizeof(float)*DIM);
            memset(gdw,0,sizeof(float)*NP*PP*LDIM);memset(gdb,0,sizeof(float)*NP*PP);
            for(int n=s;n<s+BATCH;n++)
                bl+=forward_backward(data+n*IMG_H*IMG_W*3,
                    gpw,gpb,gplg,gplb,gfw1,gfb1,gfw2,gfb2,gdw,gdb);
            bl/=BATCH;el+=bl;model.step++;

            float lr=3e-4f*(1.0f-(float)ep/(float)EPOCHS*0.5f);
            float bc1=1-powf(.9f,model.step),bc2=1-powf(.999f,model.step);

            /* Adam decoder */
            for(int i=0;i<NP*PP*LDIM;i++){
                float gi=gdw[i]/BATCH;
                model.dwm[i]=.9*model.dwm[i]+.1*gi;
                model.dwv[i]=.999*model.dwv[i]+.001*gi*gi;
                model.dw[i]-=lr*(model.dwm[i]/bc1)/(sqrtf(model.dwv[i]/bc2)+1e-8f);
            }
            for(int i=0;i<NP*PP;i++){
                float gi=gdb[i]/BATCH;
                model.dbm[i]=.9*model.dbm[i]+.1*gi;
                model.dbv[i]=.999*model.dbv[i]+.001*gi*gi;
                model.db[i]-=lr*(model.dbm[i]/bc1)/(sqrtf(model.dbv[i]/bc2)+1e-8f);
            }
            /* Adam encoder */
            for(int i=0;i<PP*DIM;i++){
                float gi=gpw[i]/BATCH;
                model.pwm[i]=.9*model.pwm[i]+.1*gi;
                model.pwv[i]=.999*model.pwv[i]+.001*gi*gi;
                model.pw[i]-=lr*(model.pwm[i]/bc1)/(sqrtf(model.pwv[i]/bc2)+1e-8f);
            }
            for(int i=0;i<DIM;i++){
                float gi=gpw?0:0; (void)gi;
                model.pbm[i]=.9*model.pbm[i]+.1*(gpb[i]/BATCH);
                model.pbv[i]=.999*model.pbv[i]+.001*(gpb[i]/BATCH)*(gpb[i]/BATCH);
                model.pb[i]-=lr*(model.pbm[i]/bc1)/(sqrtf(model.pbv[i]/bc2)+1e-8f);
                model.plgm[i]=.9*model.plgm[i]+.1*(gplg[i]/BATCH);
                model.plgv[i]=.999*model.plgv[i]+.001*(gplg[i]/BATCH)*(gplg[i]/BATCH);
                model.plg[i]-=lr*(model.plgm[i]/bc1)/(sqrtf(model.plgv[i]/bc2)+1e-8f);
                model.plbm[i]=.9*model.plbm[i]+.1*(gplb[i]/BATCH);
                model.plbv[i]=.999*model.plbv[i]+.001*(gplb[i]/BATCH)*(gplb[i]/BATCH);
                model.plb[i]-=lr*(model.plbm[i]/bc1)/(sqrtf(model.plbv[i]/bc2)+1e-8f);
            }
            for(int l=0;l<LVL;l++){
                for(int i=0;i<DIM*FH;i++){
                    float gi=gfw1[i]/BATCH;
                    model.fwm[l][i]=.9*model.fwm[l][i]+.1*gi;
                    model.fwv[l][i]=.999*model.fwv[l][i]+.001*gi*gi;
                    model.fw1[l][i]-=lr*(model.fwm[l][i]/bc1)/(sqrtf(model.fwv[l][i]/bc2)+1e-8f);
                }
                for(int i=0;i<FH;i++){
                    float gi=gfb1[i]/BATCH;
                    model.fbm[l][i]=.9*model.fbm[l][i]+.1*gi;
                    model.fbv[l][i]=.999*model.fbv[l][i]+0.001*gi*gi;
                    model.fb1[l][i]-=lr*(model.fbm[l][i]/bc1)/(sqrtf(model.fbv[l][i]/bc2)+1e-8f);
                }
                for(int i=0;i<FH*DIM;i++){
                    float gi=gfw2[i]/BATCH;
                    model.fwm2[l][i]=.9*model.fwm2[l][i]+.1*gi;
                    model.fwv2[l][i]=.999*model.fwv2[l][i]+0.001*gi*gi;
                    model.fw2[l][i]-=lr*(model.fwm2[l][i]/bc1)/(sqrtf(model.fwv2[l][i]/bc2)+1e-8f);
                }
                for(int i=0;i<DIM;i++){
                    float gi=gfb2[i]/BATCH;
                    model.fbm2[l][i]=.9*model.fbm2[l][i]+.1*gi;
                    model.fbv2[l][i]=.999*model.fbv2[l][i]+0.001*gi*gi;
                    model.fb2[l][i]-=lr*(model.fbm2[l][i]/bc1)/(sqrtf(model.fbv2[l][i]/bc2)+1e-8f);
                }
            }
        }
        el/=nb;
        float psnr=el<1e-10?100:10*log10f(1/el);
        printf("Epoch %3d/%d | Loss: %.6f | PSNR: %.1f dB\n",ep+1,EPOCHS,el,psnr);
        fprintf(log,"%d,%.6f,%.2f\n",ep+1,el,psnr);
        if((ep+1)%25==0){
            char path[256];snprintf(path,sizeof(path),"/home/wubu/WuBuMath/output/ckpt_e%03d.bin",ep+1);
            FILE*f=fopen(path,"wb");fwrite(&model,sizeof(Model),1,f);fclose(f);
            printf("  -> Saved: %s\n",path);
        }
    }
    fclose(log);

    FILE*f=fopen("/home/wubu/WuBuMath/output/wubu_trained_final.bin","wb");
    fwrite(&model,sizeof(Model),1,f);fclose(f);
    printf("\nDone. Model: output/wubu_trained_final.bin\n");
    printf("Log: output/train_log.csv\n");

    free(data);
    free(gpw);free(gpb);free(gplg);free(gplb);
    free(gfw1);free(gfb1);free(gfw2);free(gfb2);
    free(gdw);free(gdb);
    return 0;
}
