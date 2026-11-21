#include "wubu_eval.h"
#include <math.h>
double wubu_accuracy(const int *pred, const int *truth, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) if (pred[i] == truth[i]) correct++;
    return (double)correct / (n > 0 ? n : 1);
}
double wubu_perplexity(const double *log_probs, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += log_probs[i];
    return exp(-s / (n > 0 ? n : 1));
}
double wubu_f1_score(int tp, int fp, int fn) {
    double prec = (tp+fp) > 0 ? (double)tp/(tp+fp) : 0;
    double rec = (tp+fn) > 0 ? (double)tp/(tp+fn) : 0;
    if (prec+rec == 0) return 0;
    return 2*prec*rec/(prec+rec);
}