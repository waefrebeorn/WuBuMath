/* wubu_config.h */
#ifndef WUBU_CONFIG_H
#define WUBU_CONFIG_H
#include <stddef.h>
/* Simple key-value config (string values) */
typedef struct { char key[64]; char val[256]; } wubu_config_entry;
typedef struct { wubu_config_entry *entries; int n; int cap; } wubu_config;
void wubu_config_init(wubu_config *c);
int wubu_config_set(wubu_config *c, const char *key, const char *val);
const char *wubu_config_get(wubu_config *c, const char *key);
void wubu_config_free(wubu_config *c);
#endif