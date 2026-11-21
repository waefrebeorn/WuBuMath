#include "wubu_profiler.h"
#include <stdio.h>
void wubu_profile_record(wubu_profile_entry *e, double dt) {
    e->total_time += dt;
    e->calls++;
}
void wubu_profile_print(wubu_profile_entry *entries, int n) {
    for (int i = 0; i < n; i++) {
        printf("%s: %.4f s (%d calls, %.4f avg)\n",
            entries[i].name, entries[i].total_time, entries[i].calls,
            entries[i].calls > 0 ? entries[i].total_time/entries[i].calls : 0);
    }
}