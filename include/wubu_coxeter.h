/* wubu_coxeter.h */
#ifndef WUBU_COXETER_H
#define WUBU_COXETER_H
#include <stddef.h>
/* Coxeter group: length of a word given simple reflections (reduced word check).
   Simplified: returns word length. */
int wubu_coxeter_length(const int *word, int n);
#endif