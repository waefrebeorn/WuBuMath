#include "wubu_checkpoint.h"
#include <stdio.h>
void wubu_checkpoint_begin(void) { /* marker */ }
void wubu_checkpoint_end(void) { /* marker */ }
int wubu_save_weights(const char *path, const double *w, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t written = fwrite(w, sizeof(double), n, f);
    fclose(f);
    return (written == n) ? 0 : -1;
}
int wubu_load_weights(const char *path, double *w, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t read = fread(w, sizeof(double), n, f);
    fclose(f);
    return (read == n) ? 0 : -1;
}