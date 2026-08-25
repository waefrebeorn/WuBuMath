/*
 * wubu_sweeptime.c -- GAP-A006: Sweep time-axis model for the beam canvas
 *
 * The canvas is a strip swept over time; each column corresponds to a
 * time instant. The t coordinate maps linearly: t = x / width * duration.
 * This gives the codec a temporal coordinate for every spatial position.
 */
#include "wubu_sweeptime.h"

float wubu_st_x_to_t(int x,int width,float duration){
    return (float)x/(float)width*duration;
}

int wubu_st_t_to_x(float t,float duration,int width){
    if(duration<=0)return 0;
    int x=(int)(t/duration*width);
    if(x<0)x=0;if(x>=width)x=width-1;
    return x;
}

/* which frame does time t fall in at fps? */
int wubu_st_t_to_frame(float t,float fps){
    if(fps<=0)return 0;
    return (int)(t*fps);
}
