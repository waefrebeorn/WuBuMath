#include "wubu_config.h"
#include <stdlib.h>
#include <string.h>
void wubu_config_init(wubu_config *c) { c->entries = NULL; c->n = 0; c->cap = 0; }
int wubu_config_set(wubu_config *c, const char *key, const char *val) {
    for (int i = 0; i < c->n; i++) {
        if (strcmp(c->entries[i].key, key) == 0) {
            strncpy(c->entries[i].val, val, 255);
            return 0;
        }
    }
    if (c->n == c->cap) {
        c->cap = c->cap ? c->cap*2 : 16;
        c->entries = realloc(c->entries, c->cap*sizeof(wubu_config_entry));
    }
    strncpy(c->entries[c->n].key, key, 63);
    strncpy(c->entries[c->n].val, val, 255);
    c->n++;
    return 0;
}
const char *wubu_config_get(wubu_config *c, const char *key) {
    for (int i = 0; i < c->n; i++) if (strcmp(c->entries[i].key, key) == 0) return c->entries[i].val;
    return NULL;
}
void wubu_config_free(wubu_config *c) { free(c->entries); c->entries = NULL; c->n = c->cap = 0; }