/* wubu_grad_clip.h */
#ifndef WUBU_GRAD_CLIP_H
#define WUBU_GRAD_CLIP_H
#include <stddef.h>
/* Clip gradients by global norm. Returns the original norm. */
double wubu_grad_clip_norm(double *grad, size_t n, double max_norm);
/* Clip gradients by value */
void wubu_grad_clip_value(double *grad, size_t n, double max_val);
#endif