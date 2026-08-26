/*
 * wubu_tools.c -- GROUP 21: Real-world video tools
 *
 * G21.01: Transcode pipeline orchestration
 * G21.02: Thumbnail extraction from bitstream
 * G21.03: Keyframe-only fast decode
 * G21.04: Segment extraction without re-encoding
 * G21.06: Resolution scaling with proper filtering
 */
#define M_PI 3.14159265358979f
#include "wubu_tools.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== G21.01: Transcode pipeline ===== */

int wubu_transcode(const char* input,const char* output,
                    const WubuTranscodeParams* params){
    char cmd[2048];
    snprintf(cmd,sizeof(cmd),
        "ffmpeg -y -i \"%s\""
        " -vf \"scale=%d:%d:flags=%s\""
        " -c:v libx264 -preset %s -crf %d"
        " -pix_fmt yuv420p -an \"%s\" 2>/dev/null",
        input,
        params->out_width,params->out_height,
        params->filter_quality==WUBU_FQ_FAST?"bilinear":"lanczos",
        params->preset?params->preset:"medium",
        params->crf,
        output);
    return system(cmd)==0?0:-1;
}

/* ===== G21.02: Thumbnail extraction ===== */

/* extract a thumbnail at a given timestamp as PPM */
int wubu_thumbnail(const char* input,float timestamp_sec,
                    int width,int height,const char* out_ppm){
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),
        "ffmpeg -y -ss %.2f -i \"%s\" -vframes 1"
        " -vf \"scale=%d:%d\" -pix_fmt rgb24 \"%s\" 2>/dev/null",
        timestamp_sec,input,width,height,out_ppm);
    return system(cmd)==0?0:-1;
}

/* generate contact sheet (grid of thumbnails) */
int wubu_contact_sheet(const char* input,int cols,int rows,
                        int thumb_w,int thumb_h,const char* out_jpg){
    /* get duration first */
    char cmd[1536];
    snprintf(cmd,sizeof(cmd),
        "ffmpeg -y -i \"%s\" -vf \"select='not(mod(n,%d))',"
        "scale=%d:%d,tile=%dx%d' -frames:v 1 \"%s\" 2>/dev/null",
        input,(cols*rows),thumb_w,thumb_h,cols,rows,out_jpg);
    return system(cmd)==0?0:-1;
}

/* ===== G21.03: Keyframe-only decode ===== */

/* extract only I-frames (fast seek + minimal decode) */
int wubu_extract_keyframes(const char* input,const char* out_dir){
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),
        "ffmpeg -y -i \"%s\" -vf \"select=eq(pict_type\\,I)\""
        " -vsync vfr \"%s/keyframe_%%04d.png\" 2>/dev/null",
        input,out_dir);
    return system(cmd)==0?0:-1;
}

/* ===== G21.04: Segment extraction ===== */

/* extract a time segment WITHOUT re-encoding (stream copy from keyframe) */
int wubu_extract_segment(const char* input,float start_sec,float duration,
                           const char* output){
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),
        "ffmpeg -y -ss %.2f -i \"%s\" -t %.2f"
        " -c copy -avoid_negative_ts make_zero \"%s\" 2>/dev/null",
        start_sec,input,duration,output);
    return system(cmd)==0?0:-1;
}

/* ===== G21.06: Resolution scaling ===== */

/* high-quality downscale using Lanczos */
int wubu_scale_lanczos(const char* input,int out_w,int out_h,
                         const char* output){
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),
        "ffmpeg -y -i \"%s\" -vf \"scale=%d:%d:flags=lanczos\""
        " -c:v libx264 -qp 0 \"%s\" 2>/dev/null",
        input,out_w,out_h,output);
    return system(cmd)==0?0:-1;
}

/* frame rate conversion */
int wubu_change_fps(const char* input,double target_fps,const char* output){
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),
        "ffmpeg -y -i \"%s\" -r %.2f -c:v libx264 -qp 0 \"%s\" 2>/dev/null",
        input,target_fps,output);
    return system(cmd)==0?0:-1;
}

/* ===== Bitrate ladder generation for adaptive streaming ===== */

static const LadderRung default_ladder[]={
    {1920,1080,23},
    {1280,720,25},
    {854,480,27},
    {640,360,29},
    {426,240,31}
};

int wubu_generate_ladder(const char* input,const char* out_dir,
                           const LadderRung* rungs,int n_rungs){
    if(!rungs){rungs=default_ladder;n_rungs=5;}
    
    for(int i=0;i<n_rungs;i++){
        char output[512];
        snprintf(output,sizeof(output),"%s/rung_%dp.mp4",out_dir,rungs[i].height);
        
        char cmd[1024];
        snprintf(cmd,sizeof(cmd),
            "ffmpeg -y -i \"%s\" -vf \"scale=-2:%d\""
            " -c:v libx264 -preset fast -crf %d"
            " -pix_fmt yuv420p -an \"%s\" 2>/dev/null",
            input,rungs[i].height,rungs[i].crf,output);
        
        if(system(cmd)!=0)return i; /* return which rung failed */
    }
    return n_rungs; /* all succeeded */
}

/* generate HLS playlist for adaptive streaming */
int wubu_generate_hls(const char* input,const char* out_dir,int segment_sec){
    char mkdir_cmd[256];
    snprintf(mkdir_cmd,256,"mkdir -p %s",out_dir);
    system(mkdir_cmd);
    
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),
        "ffmpeg -y -i \"%s\" -c:v libx264 -preset fast -crf 25"
        " -hls_time %d -hls_playlist_type vod"
        " -hls_segment_filename \"%s/seg_%%03d.ts\" \"%s/playlist.m3u8\" 2>/dev/null",
        input,segment_sec,out_dir,out_dir);
    return system(cmd)==0?0:-1;
}
