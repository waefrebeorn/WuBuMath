/* wubu_checkpoint.h */
#ifndef WUBU_CHECKPOINT_H
#define WUBU_CHECKPOINT_H
#include <stddef.h>
/* Gradient checkpointing: recompute forward during backward to save memory.
   Simplified: just a marker function. */
void wubu_checkpoint_begin(void);
void wubu_checkpoint_end(void);
/* Save/load weights to binary file */
int wubu_save_weights(const char *path, const double *w, size_t n);
int wubu_load_weights(const char *path, double *w, size_t n);
#endif