//
// Error-handling functions from CSAPP-3e
//

#ifndef SERVER_ERROR_H
#define SERVER_ERROR_H
void unix_error(char *msg);
void gai_error(int code, char *msg);
void app_error(char *msg);
#endif //TCP_ERROR_H
