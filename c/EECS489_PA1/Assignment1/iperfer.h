#ifndef EECS489_PA1_IPERFER_H
#define EECS489_PA1_IPERFER_H
#define MAX_BUF 1000
#define ACK_LENGTH 4

#include <stdlib.h>
#include <stdio.h>
void error(const char *message);
void server(const char *port);
void client (const char *host, const char *port, const char *time);

#endif
