/* wubu_tokenizer.h */
#ifndef WUBU_TOKENIZER_H
#define WUBU_TOKENIZER_H
#include <stddef.h>
/* Simple character-level tokenizer: returns number of tokens */
int wubu_tokenize(const char *text, int *tokens, int max_tokens);
/* Detokenize: reconstruct text from tokens */
int wubu_detokenize(const int *tokens, int n, char *text, int max_len);
#endif