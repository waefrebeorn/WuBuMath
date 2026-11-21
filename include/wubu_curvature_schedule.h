/* wubu_curvature_schedule.h */
#ifndef WUBU_CURVATURE_SCHEDULE_H
#define WUBU_CURVATURE_SCHEDULE_H
#include <stddef.h>
/* Curvature annealing: warmup then decay. */
double wubu_curvature_schedule(int step, int warmup, double K_init, double K_min, double K_max);
#endif