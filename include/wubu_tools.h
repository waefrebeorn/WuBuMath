/* GROUP 21: Real-world video tools */
#ifndef WUBU_TOOLS_H
#define WUBU_TOOLS_H
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {WUBU_FQ_FAST,WUBU_FQ_HIGH} wubu_filter_quality_t;

typedef struct {
    int out_width,out_height,crf;
    const char* preset;
    wubu_filter_quality_t filter_quality;
} WubuTranscodeParams;

int wubu_transcode(const char* input,const char* output,
                    const WubuTranscodeParams* params);
int wubu_thumbnail(const char* input,float timestamp_sec,
                    int width,int height,const char* out_ppm);
int wubu_contact_sheet(const char* input,int cols,int rows,
                        int thumb_w,int thumb_h,const char* out_jpg);
int wubu_extract_keyframes(const char* input,const char* out_dir);
int wubu_extract_segment(const char* input,float start_sec,float duration,
                           const char* output);
int wubu_scale_lanczos(const char* input,int out_w,int out_h,const char* output);
int wubu_change_fps(const char* input,double target_fps,const char* output);

typedef struct { int width,height,crf; } LadderRung;
int wubu_generate_ladder(const char* input,const char* out_dir,
                           const LadderRung* rungs,int n_rungs);
int wubu_generate_hls(const char* input,const char* out_dir,int segment_sec);
#ifdef __cplusplus
}
#endif
#endif
