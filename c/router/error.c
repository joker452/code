#include "error.h"
void error(const char *message) {
    fprintf(stderr, "%s", message);
    exit(EXIT_FAILURE);
}