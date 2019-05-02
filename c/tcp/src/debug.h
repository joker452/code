#ifndef DEBUG_H
#define DEBUG_H
#ifdef NDEBUG
#define debug(M, ...)
#else
#define debug(M, ...) fprintf(stderr, "[DEBUG message] %s:%d: " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif
#endif

