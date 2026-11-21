#define _GNU_SOURCE
#include "wubu_lr_schedule.h"
#include <math.h>
double wubu_lr_cosine(int step, int total, double lr_max, double lr_min) {
    double progress = (double)step / (double)total;
    return lr_min + 0.5*(lr_max-lr_min)*(1.0+cos(M_PI*progress));
}
double wubu_lr_warmup(int step, int warmup, double lr_max) {
    if (step < warmup) return lr_max * (double)step / (double)warmup;
    return lr_max;
}
double wubu_lr_step(int step, double lr0, int step_size, double gamma) {
    int n = step / step_size;
    double lr = lr0;
    for (int i = 0; i < n; i++) lr *= gamma;
    return lr;
}