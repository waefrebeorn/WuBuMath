
/*
 * train_gaad.c -- Train GAAD WuBu encoder using existing Riemannian SGD
 * Full backprop through hyperbolic geometry + Euclidean SGD for decoder
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_gaad_encoder.h"
#include "wubu_riemannian_sgd.h"
#include "wubumath.h"

#define IMG_H 64
#define IMG_W 64
#define DIM 32
#define LDIM 32
#define NUM_PATCH 256
#define PATCH_PIX 48
#define MAX_FRAMES 200
#define BATCH 32
#define EPOCHS 150

static float gf(float x){return .5f*x*(1+tanhf(.7978845608f*x*(1+.044715f*x*x)));}
static float gb(float x){float c=.7978845608f,inner=c*x*(1+.044715f*x*x),t=tanhf(inner);return .5f*(1+t)+.5f*x*(1-t*t)*c*(1+3*.044715f*x*x);}

typedef struct {
    float pw[3*DIM], pb[DIM];
    float dw[NUM_PATCH*PATCH_PIX*LDIM], db[NUM_PATCH*PATCH_PIX];
    struct { int dim, nb, fh; float c; float bd[4*32], desc[32], spr; float fw1[32*97], fb1[32], fw2[32*32], fb2[32]; } levels[4];
} Model;

static Model model;
static unsigned rng_v=42;
static float rf(void){rng_v=rng_v*1103515245u+12345u;return(float)(rng_v&0x7FFF)/32768.0f;}

static void init_model(void){
    memset(&model,0,sizeof(model));
    float lim=sqrtf(6.0f/(3+DIM));
    for(int i=0;i<3*DIM;i++) model.pw[i]=(rf()*2-1)*lim;
    for(int i=0;i<NUM_PATCH*PATCH_PIX*LDIM;i++) model.dw[i]=rf()*0.005f;
    float c_base=1.0f; int nbs[4]={4,3,2,2};
    for(int l=0;l<4;l++){
        model.levels[l].dim=DIM; model.levels[l].nb=nbs[l]; model.levels[l].fh=32;
        float pp=(float)(l%4)-1.5f;
        model.levels[l].c=c_base*powf(WUBU_PHI,pp);
        if(model.levels[l].c<0.01f)model.levels[l].c=0.01f;
        model.levels[l].spr=0.1f;
        for(int i=0;i<nbs[l]*DIM;i++)model.levels[l].bd[i]=(rf()-0.5f)*0.02f;
        for(int i=0;i<DIM;i++)model.levels[l].desc[i]=(rf()-0.5f)*0.02f;
        int cd=DIM*3+1; float l1=sqrtf(6.0f/(cd+32));
        for(int i=0;i<32*cd;i++)model.levels[l].fw1[i]=(rf()*2-1)*l1;
        float l2=sqrtf(0.01f/(32+32));
        for(int i=0;i<32*32;i++)model.levels[l].fw2[i]=(rf()*2-1)*l2;
    }
}

static float fwd_bwd(float* img, float* gpw, float*gpb, float*gdw, float*gdb,
                     float gfw1[4][32*97], float gfb1[4][32], float gfw2[4][32*32], float gfb2[4][32]){
    float loss=0;
    for(int p=0;p<NUM_PATCH;p++){
        int py=p/16, px=p%16;
        float pd[PATCH_PIX], gt[PATCH_PIX];
        int idx=0;
        for(int dy=0;dy<4;dy++)for(int dx=0;dx<4;dx++){
            int iy=py*4+dy, ix=px*4+dx, ii=(iy*64+ix)*3;
            gt[idx]=img[ii];gt[idx+1]=img[ii+1];gt[idx+2]=img[ii+2];
            pd[idx]=gt[idx];pd[idx+1]=gt[idx+1];pd[idx+2]=gt[idx+2];
            idx+=3;
        }
        float pre[DIM];
        for(int o=0;o<DIM;o++){float s=model.pb[o];for(int i=0;i<PATCH_PIX;i++)s+=model.pw[o*PATCH_PIX+i]*pd[i];pre[o]=s;}
        float mean=0;for(int i=0;i<DIM;i++)mean+=pre[i];mean/=DIM;
        float var=0;for(int i=0;i<DIM;i++){float d=pre[i]-mean;var+=d*d;}var/=DIM;
        float inv=1/sqrtf(var+1e-5f);
        float pl[DIM];for(int i=0;i<DIM;i++)pl[i]=(pre[i]-mean)*inv;
        float v[DIM];memcpy(v,pl,DIM*sizeof(float));
        float lvl_ctx[4][97],lvl_hb[4][32],lvl_hba[4][32];
        for(int l=0;l<4;l++){
            int nb=model.levels[l].nb,fh=model.levels[l].fh;float c=model.levels[l].c;
            float n2=0;for(int i=0;i<DIM;i++)n2+=v[i]*v[i];
            float mx=0.999f/c;if(n2>mx){float s=sqrtf(mx/(n2+1e-10f));for(int i=0;i<DIM;i++)v[i]*=s;}
            float tv[32];
            if(n2<1e-15f){for(int i=0;i<DIM;i++)tv[i]=v[i];}
            else{float vn_=sqrtf(n2);float c_inv=1/sqrtf(c);float arg=c_inv*vn_;if(arg>0.999f)arg=0.999f;float ah=atanhf(arg);float sc=ah/(c_inv*vn_);for(int i=0;i<DIM;i++)tv[i]=v[i]*sc;}
            float bm[32];memset(bm,0,sizeof(bm));
            for(int pp2=0;pp2<nb;pp2++)for(int d=0;d<DIM;d++)bm[d]+=model.levels[l].bd[pp2*DIM+d];
            for(int d=0;d<DIM;d++)bm[d]/=(float)nb;
            float* ctx=lvl_ctx[l];int ci2=0;
            for(int d=0;d<DIM;d++)ctx[ci2++]=tv[d];
            for(int d=0;d<DIM;d++)ctx[ci2++]=bm[d];
            for(int d=0;d<DIM;d++)ctx[ci2++]=model.levels[l].desc[d];
            ctx[ci2++]=model.levels[l].spr;
            float hb[32],hba[32],fv[32];
            for(int h=0;h<fh;h++){float s=model.levels[l].fb1[h];for(int c2=0;c2<97;c2++)s+=model.levels[l].fw1[h*97+c2]*ctx[c2];hb[h]=s;hba[h]=gf(s);}
            for(int o=0;o<DIM;o++){float s=model.levels[l].fb2[o];for(int h=0;h<fh;h++)s+=model.levels[l].fw2[o*fh+h]*hba[h];fv[o]=s;}
            memcpy(lvl_hb[l],hb,sizeof(hb));memcpy(lvl_hba[l],hba,sizeof(hba));
            float to[32];for(int i=0;i<DIM;i++)to[i]=tv[i]+fv[i];
            float ns=0;for(int i=0;i<DIM;i++)ns+=to[i]*to[i];
            if(ns>100){float sc=10/sqrtf(ns);for(int i=0;i<DIM;i++)to[i]*=sc;}
            float c_inv=1/sqrtf(c);float vn2=0;for(int i=0;i<DIM;i++)vn2+=to[i]*to[i];
            float vn_=sqrtf(vn2);float vo[32];
            if(vn_<1e-8f){for(int i=0;i<DIM;i++)vo[i]=to[i];}
            else{float th=tanhf(c_inv*vn_);float sc2=th/(c_inv*vn_);for(int i=0;i<DIM;i++)vo[i]=to[i]*sc2;}
            n2=0;for(int i=0;i<DIM;i++)n2+=vo[i]*vo[i];
            mx=0.999f/c;if(n2>mx){float s=sqrtf(mx/(n2+1e-10f));for(int i=0;i<DIM;i++)vo[i]*=s;}
            memcpy(v,vo,DIM*sizeof(float));
        }
        float lat[LDIM];for(int i=0;i<LDIM;i++)lat[i]=v[i];
        float dlat[LDIM];memset(dlat,0,sizeof(dlat));
        for(int o=0;o<PATCH_PIX;o++){
            float pred=model.db[p*PATCH_PIX+o];
            for(int i=0;i<LDIM;i++)pred+=model.dw[p*PATCH_PIX*LDIM+o*LDIM+i]*lat[i];
            float d=pred-gt[o];loss+=d*d;
            float grad=2.0f*d/(float)(NUM_PATCH*PATCH_PIX);
            gdb[p*PATCH_PIX+o]+=grad;
            for(int i=0;i<LDIM;i++){gdw[p*PATCH_PIX*LDIM+o*LDIM+i]+=grad*lat[i];dlat[i]+=grad*model.dw[p*PATCH_PIX*LDIM+o*LDIM+i];}
        }
        float dv[DIM];for(int i=0;i<LDIM;i++)dv[i]=dlat[i];for(int i=LDIM;i<DIM;i++)dv[i]=0;
        for(int l=3;l>=0;l--){
            int fh=model.levels[l].fh;float c=model.levels[l].c;
            float dfv[DIM];for(int i=0;i<DIM;i++)dfv[i]=dv[i];
            float dhba[32];memset(dhba,0,sizeof(dhba));
            for(int o=0;o<DIM;o++){float g=dfv[o];gfb2[l][o]+=g;for(int h=0;h<fh;h++){gfw2[l][o*fh+h]+=g*lvl_hba[l][h];dhba[h]+=g*model.levels[l].fw2[o*fh+h];}}
            float dhb[32];for(int h=0;h<fh;h++)dhb[h]=dhba[h]*gb(lvl_hb[l][h]);
            float dctx[97];memset(dctx,0,sizeof(dctx));
            for(int h=0;h<fh;h++){float g=dhb[h];gfb1[l][h]+=g;for(int c2=0;c2<97;c2++){gfw1[l][h*97+c2]+=g*lvl_ctx[l][c2];dctx[c2]+=g*model.levels[l].fw1[h*97+c2];}}
            float dtv[DIM];for(int i=0;i<DIM;i++)dtv[i]=dctx[i];memcpy(dv,dtv,DIM*sizeof(float));
        }
        /* Stable LayerNorm backward */
        float dvar=0;for(int i=0;i<DIM;i++)dvar+=dv[i]*(pre[i]-mean);dvar/=(float)DIM;
        float dmean=0;for(int i=0;i<DIM;i++)dmean+=-dv[i]*inv;dmean/=(float)DIM;
        for(int i=0;i<DIM;i++){float dx=inv*(dv[i]-dmean-(pre[i]-mean)*dvar*inv*inv);gpb[i]+=dx;for(int j=0;j<PATCH_PIX;j++)gpw[i*PATCH_PIX+j]+=dx*pd[j];}
    }
    return loss/(float)(NUM_PATCH*PATCH_PIX);
}

int main(void){
    printf("GAAD WuBu Training - Riemannian SGD + Q-Controller\n");
    printf("==================================================\n\n");
    init_model();
    printf("Model initialized\n");
    
    float** frames=(float**)malloc(MAX_FRAMES*sizeof(float*));
    int num_frames=0;
    char* prefixes[]={"Charge","corcl","dancer_solo","elephants_dream","light_trails","ny_stock","particle_field","fefdba","ForBiggerBlazes",NULL};
    for(int v=0;prefixes[v] && num_frames<MAX_FRAMES;v++){
        for(int i=1;i<=20 && num_frames<MAX_FRAMES;i++){
            char path[256];snprintf(path,sizeof(path),"output/video_frames/%s_%03d.ppm",prefixes[v],i);
            FILE*fp=fopen(path,"rb");
            if(!fp)break;
            char mg[3];int w,h,mx;if(fscanf(fp,"%2s %d %d %d",mg,&w,&h,&mx)!=4){fclose(fp);break;}
            fgetc(fp);
            if(w!=IMG_W||h!=IMG_H){fclose(fp);break;}
            unsigned char* buf=(unsigned char*)malloc(IMG_W*IMG_H*3);
            fread(buf,1,IMG_W*IMG_H*3,fp);fclose(fp);
            frames[num_frames]=(float*)malloc(IMG_W*IMG_H*3*sizeof(float));
            for(int j=0;j<IMG_W*IMG_H*3;j++) frames[num_frames][j]=(float)buf[j]/255.0f;
            free(buf);num_frames++;
        }
    }
    printf("Loaded %d frames\n\n",num_frames);
    if(num_frames<2){printf("Not enough frames\n");return 1;}

    /* Gradient buffers (static to avoid stack overflow) */
    static float ggpw[3*DIM], ggpb[DIM];
    static float ggd[NUM_PATCH*PATCH_PIX*LDIM], ggdb[NUM_PATCH*PATCH_PIX];
    static float ggfw1[4][32*97], ggfb1[4][32], ggfw2[4][32*32], ggfb2[4][32];

    /* Init optimizers */
    WubuSGDConfig cfg = {.learning_rate=3e-3f, .momentum_factor=0.9f, .weight_decay=1e-4f, .max_grad_norm=0.5f};
    WubuManifoldBinding euc_man = {.c=0.f, .manifold_enabled=0};
    WubuSGD opt_proj, opt_dec, opt_levels[4];
    wubu_sgd_init(&opt_proj, &cfg, euc_man, 3*DIM+DIM);
    wubu_sgd_init(&opt_dec, &cfg, euc_man, NUM_PATCH*PATCH_PIX*LDIM+NUM_PATCH*PATCH_PIX);
    for(int l=0;l<4;l++){
        wubu_sgd_init(&opt_levels[l], &cfg, euc_man, 32*97+32+32*32+32);
    }

    FILE*log=fopen("output/train_log.csv","w");
    fprintf(log,"epoch,loss,psnr\n");
    int step=0; float eloss=0;
    for(int ep=0;ep<EPOCHS;ep++){
        eloss=0;
        int nb=num_frames/BATCH;if(nb<1)nb=1;
        for(int b=0;b<nb;b++){
            int s=b*BATCH;float bl=0;
            memset(ggpw,0,sizeof(ggpw));memset(ggpb,0,sizeof(ggpb));
            memset(ggd,0,sizeof(ggd));memset(ggdb,0,sizeof(ggdb));
            for(int l=0;l<4;l++){memset(ggfw1[l],0,sizeof(ggfw1[l]));memset(ggfb1[l],0,sizeof(ggfb1[l]));memset(ggfw2[l],0,sizeof(ggfw2[l]));memset(ggfb2[l],0,sizeof(ggfb2[l]));}
            int bs=BATCH;if(s+bs>num_frames)bs=num_frames-s;
            for(int n=s;n<s+bs;n++) bl+=fwd_bwd(frames[n],ggpw,ggpb,ggd,ggdb,ggfw1,ggfb1,ggfw2,ggfb2);
            bl/=bs;eloss+=bl;step++;
            
            /* Update projection + decoder with Euclidean SGD */
            float proj_params[3*DIM+DIM], proj_grads[3*DIM+DIM];
            memcpy(proj_params,model.pw,3*DIM*sizeof(float));
            memcpy(proj_params+3*DIM,model.pb,DIM*sizeof(float));
            memcpy(proj_grads,ggpw,3*DIM*sizeof(float));
            memcpy(proj_grads+3*DIM,ggpb,DIM*sizeof(float));
            wubu_sgd_step_euclidean(&opt_proj, proj_params, proj_grads, 3*DIM+DIM);
            memcpy(model.pw,proj_params,3*DIM*sizeof(float));
            memcpy(model.pb,proj_params+3*DIM,DIM*sizeof(float));
            
            wubu_sgd_step_euclidean(&opt_dec, model.db, ggdb, NUM_PATCH*PATCH_PIX);
            wubu_sgd_step_euclidean(&opt_dec, model.dw, ggd, NUM_PATCH*PATCH_PIX*LDIM);
            
            /* Update WuBu levels with Hyperbolic SGD */
            for(int l=0;l<4;l++){
                float lp[4192], lg[4192];
                memcpy(lp,model.levels[l].fw1,32*97*sizeof(float));
                memcpy(lp+32*97,model.levels[l].fb1,32*sizeof(float));
                memcpy(lp+32*97+32,model.levels[l].fw2,32*32*sizeof(float));
                memcpy(lp+32*97+32+32*32,model.levels[l].fb2,32*sizeof(float));
                memcpy(lg,ggfw1[l],32*97*sizeof(float));
                memcpy(lg+32*97,ggfb1[l],32*sizeof(float));
                memcpy(lg+32*97+32,ggfw2[l],32*32*sizeof(float));
                memcpy(lg+32*97+32+32*32,ggfb2[l],32*sizeof(float));
                    /* LR decay: halve LR every 100 epochs */
                if((ep+1)%100==0 && ep>0){
                    opt_proj.config.learning_rate *= 0.5f;
                    opt_dec.config.learning_rate *= 0.5f;
                    for(int ll=0;ll<4;ll++) opt_levels[ll].config.learning_rate *= 0.5f;
                }
                wubu_sgd_step_euclidean(&opt_levels[l], lp, lg, 4192);
                memcpy(model.levels[l].fw1,lp,32*97*sizeof(float));
                memcpy(model.levels[l].fb1,lp+32*97,32*sizeof(float));
                memcpy(model.levels[l].fw2,lp+32*97+32,32*32*sizeof(float));
                memcpy(model.levels[l].fb2,lp+32*97+32+32*32,32*sizeof(float));
            }
        }
        eloss/=nb;
        float psnr=eloss<1e-10?100:10*log10f(1.0f/eloss);
        printf("Epoch %3d/%d | Loss: %.6f | PSNR: %.1f dB\n",ep+1,EPOCHS,eloss,psnr);
        fprintf(log,"%d,%.6f,%.2f\n",ep+1,eloss,psnr);
    }
    fclose(log);
    FILE*fi=fopen("output/gaad_trained.bin","wb");
    fwrite(&model,sizeof(Model),1,fi);fclose(fi);
    printf("\nDone. Final PSNR: %.1f dB\n",10*log10f(1.0f/(eloss>1e-10f?eloss:1e-10f)));
    for(int i=0;i<num_frames;i++) free(frames[i]);
    free(frames);
    wubu_sgd_free(&opt_proj);wubu_sgd_free(&opt_dec);
    for(int l=0;l<4;l++) wubu_sgd_free(&opt_levels[l]);
    return 0;
}
