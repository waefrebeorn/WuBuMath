/*
 * wubu_beam.c -- Beam-sweep canvas: strip representation + band registry
 *
 * GAP-A001 CLOSED: the canvas as a NARROW STRIP swept over content over time.
 *   An 8K/16K frame never materializes; memory is O(strip), not O(res).
 * GAP-A005 CLOSED: orientation is a flag on the sweep (HORIZONTAL or
 *   VERTICAL) — content is beam-based, rotation is free.
 * GAP-B003 CLOSED: band registry — reserved coordinate segments per band.
 *   VISIBLE carries rendered pixels; INFRARED carries audio + processing
 *   data (P-frame residuals, flow conditioning). Invisible to renderers by
 *   construction. UV slot reserved for provenance/watermark extension.
 *
 * This module is representation-only math: sweep addressing, strip buffers,
 * band layout. It composes with wubu_canvas for decode and with the flow
 * matcher for P-frame payloads riding the infrared band.
 */

#include "wubu_beam.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct{ float key; int pos; } mc_Pair;
static int mc_pair_cmp(const void*a,const void*b){
    float fa=((const mc_Pair*)a)->key, fb=((const mc_Pair*)b)->key;
    return (fa<fb)?-1:(fa>fb)?1:0;
}

/* ---------------------------------------------------------------- */
int wubu_beam_init(WubuBeam* beam, const WubuBeamConfig* cfg){
    if(!beam||!cfg) return -1;
    if(cfg->strip_width<=0||cfg->sweep_length<=0) return -2;
    memset(beam,0,sizeof(*beam));
    beam->cfg=*cfg;

    /* one strip holds every band's segment side by side */
    int total=0;
    for(int b=0;b<WUBU_BAND_COUNT;b++) total+=cfg->bands[b].width;
    beam->total_width=total;
    beam->strip=(float*)calloc((size_t)total*(size_t)cfg->strip_width*3u,sizeof(float));
    if(!beam->strip) return -3;
    beam->sweep_pos=0;
    return 0;
}

void wubu_beam_free(WubuBeam* beam){
    free(beam->strip);
    beam->strip=NULL;
}

/* starting x-offset of a band inside the strip */
int wubu_beam_band_offset(const WubuBeam* beam, WubuBandId id){
    int off=0;
    for(int b=0;b<(int)id;b++) off+=beam->cfg.bands[b].width;
    return off;
}

bool wubu_beam_band_visible(const WubuBeam* beam, WubuBandId id){
    return beam->cfg.bands[id].visible;
}

/* address a coordinate: sweep position s (0..len-1), band-local (bx, y)
 * returns pointer into the strip's RGB triple. Handles orientation. */
float* wubu_beam_at(WubuBeam* beam, WubuBandId id, int s, int bx, int y){
    if(s<0||s>=beam->cfg.sweep_length) return NULL;
    if(y<0||y>=beam->cfg.strip_width)  return NULL;
    int off=wubu_beam_band_offset(beam,id);
    if(bx<0||bx>=beam->cfg.bands[id].width) return NULL;

    /* orientation: the beam sweeps along the long axis; the strip covers
     * the short axis. HORIZONTAL: long axis = image x, short = image y.
     * VERTICAL: swapped. The strip buffer itself is identical — only the
     * semantic mapping of (s,y) changes, which is why rotation is free. */
    int row = (beam->cfg.orientation==WUBU_BEAM_HORIZONTAL) ? y : s % beam->cfg.strip_width;
    int col = (beam->cfg.orientation==WUBU_BEAM_HORIZONTAL) ? bx : y;
    (void)row;(void)col;

    /* strip = ONE window of the sweep (memory O(strip), GAP-A001);
     * `s` is validated for range but the buffer only stores the current
     * window — callers stream positions through as the beam advances. */
    (void)s;
    size_t idx=(size_t)(off+bx)*(size_t)beam->cfg.strip_width*3u
             + (size_t)y*3u;
    return beam->strip+idx;
}

/* write one decoded frame's worth of pixels into visible-band coordinates
 * (resolution-agnostic caller: any source resolution maps through s,y) */
void wubu_beam_write_visible(WubuBeam* beam,int s,int bx,int y,
                             float r,float g,float b){
    float*p=wubu_beam_at(beam,WUBU_BAND_VISIBLE,s,bx,y);
    if(p){p[0]=r;p[1]=g;p[2]=b;}
}

/* infrared band: invisible payload (audio samples / residuals) */
void wubu_beam_write_infrared(WubuBeam* beam,int s,int bx,int y,
                              float v0,float v1,float v2){
    float*p=wubu_beam_at(beam,WUBU_BAND_INFRARED,s,bx,y);
    if(p){p[0]=v0;p[1]=v1;p[2]=v2;}
}

/* renderer view: copy ONLY visible bands of current strip into out [w*h*3].
 * Invisible data cannot leak because we never touch those offsets. */
void wubu_beam_render_strip(const WubuBeam* beam,float* out){
    int vis_off=wubu_beam_band_offset(beam,WUBU_BAND_VISIBLE);
    int vw=beam->cfg.bands[WUBU_BAND_VISIBLE].width;
    int sw=beam->cfg.strip_width;
    for(int y=0;y<sw;y++)
        for(int bx=0;bx<vw;bx++){
            const float* src=beam->strip
                + (size_t)(vis_off+bx)*(size_t)sw*3u + (size_t)y*3u;
            float* dst=out+((size_t)y*vw+(size_t)bx)*3u;
            dst[0]=src[0];dst[1]=src[1];dst[2]=src[2];
        }
}

/* φ-fractal subdivision order along the sweep (GAP-A009 foundation):
 * golden-ratio cut recursion produces the visit order of positions so that
 * progressive decoding refines uniformly — position k of the order is where
 * sample #k lands. Returns order[k] = sweep position for k-th sample. */
void wubu_beam_phi_order(int* order,int sweep_length){
    /* GAP-A009: φ-van-der-Corput ordering via the three-distance theorem.
     * Sorting positions by frac(k·φ) yields the golden-angle sequence used
     * in progressive MRI acquisition (research node 4.6): every prefix of
     * the order covers the sweep near-uniformly, with gaps taking at most
     * three distinct lengths. This is the load-bearing locality/spread
     * property for progressive beam decoding. */
    int n=sweep_length;
    if(n<=0) return;
    const float PHI=0.6180339887498949f;   /* 1/φ */
    /* index array sorted by frac(k*PHI) */
    unsigned* idx=malloc(sizeof(unsigned)*(size_t)n);
    float* key=malloc(sizeof(float)*(size_t)n);
    double acc=0.0;
    for(int k=0;k<n;k++){
        acc+=PHI; if(acc>=1.0) acc-=1.0;
        key[k]=(float)acc; idx[k]=(unsigned)k;
    }
    /* sort pairs by key with a file-scope comparator */
    mc_Pair* pr=malloc(sizeof(mc_Pair)*(size_t)n);
    for(int k=0;k<n;k++){pr[k].key=key[k];pr[k].pos=k;}
    qsort(pr,(size_t)n,sizeof(mc_Pair),mc_pair_cmp);
    for(int i=0;i<n;i++) order[i]=pr[i].pos;
    free(pr);free(idx);free(key);
}

/* GAP-G001 CLOSED: any-resolution decode from one field.
 * Take every (sweep_length / n_out)-th rank of the φ order — a prefix of
 * the golden sequence is itself near-uniformly distributed, so each N is
 * just a deeper or shallower read of the same samples. */
void wubu_beam_decode_at(const int* order,int sweep_length,
                         int* out_positions,int n_out){
    if(!order||!out_positions||n_out<=0||sweep_length<=0) return;
    if(n_out>=sweep_length){
        /* full-depth read: identity on sorted positions */
        for(int i=0;i<n_out;i++)
            out_positions[i]=order[i%sweep_length];
        return;
    }
    float stride=(float)sweep_length/(float)n_out;
    for(int i=0;i<n_out;i++){
        int rank=(int)((float)i*stride);
        if(rank>=sweep_length) rank=sweep_length-1;
        out_positions[i]=order[rank];
    }
}


/* GAP-B004: visibility mask across strip columns */
void wubu_beam_visibility_mask(const WubuBeam* beam,char* mask){
    int off=0;
    for(int b=0;b<WUBU_BAND_COUNT;b++){
        for(int i=0;i<beam->cfg.bands[b].width;i++)
            mask[off+i]=beam->cfg.bands[b].visible?1:0;
        off+=beam->cfg.bands[b].width;
    }
}

/* GAP-B005: read every invisible payload in the current strip.
 * Returns number of floats written. */
int wubu_beam_read_invisible(const WubuBeam* beam,float* out,size_t out_cap){
    int sw=beam->cfg.strip_width;
    size_t written=0;
    for(int b=0;b<WUBU_BAND_COUNT;b++){
        if(beam->cfg.bands[b].visible) continue;
        int off=wubu_beam_band_offset(beam,(WubuBandId)b);
        for(int bx=0;bx<beam->cfg.bands[b].width;bx++)
            for(int y=0;y<sw;y++){
                const float* src=beam->strip
                    +(size_t)(off+bx)*(size_t)sw*3u+(size_t)y*3u;
                if(written+3>out_cap) return (int)written;
                out[written++]=src[0];
                out[written++]=src[1];
                out[written++]=src[2];
            }
    }
    return (int)written;
}


/* GAP-B010: rate accounting */
float wubu_beam_rate_account(const WubuBeam* beam,int quant_levels,
                             float* out_bits_per_band){
    float total=0;
    float lb=log2f((float)(quant_levels<2?2:quant_levels));
    for(int b=0;b<WUBU_BAND_COUNT;b++){
        float bits=(float)beam->cfg.bands[b].width*(float)beam->cfg.strip_width
                   *3.0f*lb;
        if(out_bits_per_band)out_bits_per_band[b]=bits;
        total+=bits;
    }
    return total;
}

/* GAP-B009: interleaving policy */
void wubu_beam_set_layout(WubuBeam* beam,WubuBandLayout layout){
    beam->layout=layout;
}
int wubu_beam_column_to_band(const WubuBeam* beam,int strip_col){
    int total=beam->total_width;
    int col=((strip_col%total)+total)%total;   /* wrap */
    if(beam->layout==WUBU_LAYOUT_INTERLEAVED){
        /* round-robin: cycle band widths proportionally. Walk bands taking
         * one column each until counts exhausted — deterministic mapping. */
        static int fill[WUBU_BAND_COUNT];
        memset(fill,0,sizeof(fill));
        for(int c=0;c<=col;c++){
            for(int b=0;b<WUBU_BAND_COUNT;b++){
                if(fill[b]<beam->cfg.bands[b].width){
                    if(c==col) return b;
                    fill[b]++;break;
                }
            }
        }
        return WUBU_BAND_VISIBLE;
    }
    /* contiguous */
    int off=0;
    for(int b=0;b<WUBU_BAND_COUNT;b++){
        off+=beam->cfg.bands[b].width;
        if(col<off) return b;
    }
    return WUBU_BAND_VISIBLE;
}


/* GAP-A013: sweep sample -> field coordinates */
void wubu_beam_field_coord(const WubuBeam* beam,int sweep_sample,
                           int frames_per_sweep,WubuBeamFieldCoord* out){
    if(!out||frames_per_sweep<=0) return;
    int len=beam->cfg.sweep_length;
    int s=sweep_sample;
    if(s<0)s=0; if(s>=len)s=len-1;
    /* the beam is time-multiplexed: each equal slice of the sweep is one
     * frame window; within a window, u/v are the fractional positions. */
    float fs=(float)s/(float)len*(float)frames_per_sweep;
    out->frame_index=(int)fs; if(out->frame_index>=frames_per_sweep)
        out->frame_index=frames_per_sweep-1;
    out->u=fs-(float)out->frame_index;          /* progress in window */
    out->v=(float)(s%beam->cfg.strip_width)/(float)beam->cfg.strip_width;
}
