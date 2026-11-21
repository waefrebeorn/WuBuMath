/* wubu_profiler.h */
#ifndef WUBU_PROFILER_H
#define WUBU_PROFILER_H
#include <stddef.h>
/* Simple profiler: record timings */
typedef struct { char name[64]; double total_time; int calls; } wubu_profile_entry;
void wubu_profile_record(wubu_profile_entry *e, double dt);
void wubu_profile_print(wubu_profile_entry *entries, int n);
#endif