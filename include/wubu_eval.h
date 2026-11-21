/* wubu_eval.h */
#ifndef WUBU_EVAL_H
#define WUBU_EVAL_H
#include <stddef.h>
/* Evaluation metrics */
double wubu_accuracy(const int *pred, const int *truth, int n);
double wubu_perplexity(const double *log_probs, int n);
double wubu_f1_score(int tp, int fp, int fn);
#endif