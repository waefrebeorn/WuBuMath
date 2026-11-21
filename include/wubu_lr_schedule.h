/* wubu_lr_schedule.h */
#ifndef WUBU_LR_SCHEDULE_H
#define WUBU_LR_SCHEDULE_H
/* Learning rate schedulers */
double wubu_lr_cosine(int step, int total, double lr_max, double lr_min);
double wubu_lr_warmup(int step, int warmup, double lr_max);
double wubu_lr_step(int step, double lr0, int step_size, double gamma);
#endif