#include "wubu_tokenizer.h"
#include <string.h>
int wubu_tokenize(const char *text, int *tokens, int max_tokens) {
    int n = 0;
    for (const char *p = text; *p && n < max_tokens; p++) tokens[n++] = (unsigned char)*p;
    return n;
}
int wubu_detokenize(const int *tokens, int n, char *text, int max_len) {
    int pos = 0;
    for (int i = 0; i < n && pos < max_len-1; i++) text[pos++] = (char)tokens[i];
    text[pos] = '\0';
    return pos;
}