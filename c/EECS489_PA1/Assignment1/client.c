#include <sys/types.h> /* send, recv */
#include <sys/socket.h> /* send, recv */
#include <netdb.h>   /* getaddrinfo */
#include <unistd.h>  /* sleep */
#include <time.h>    /* clock_gettime */
#include <string.h>  /* memset, memcpy */
#include <pthread.h> /* pthread_create, pthread_join */
#include "iperfer.h"

static char data[MAX_BUF];
static char buffer[ACK_LENGTH];

struct parameter {
    int fsd;
    int sent;
};

void *send_data(void *param)
{
    struct parameter *p = (struct parameter*)param;
    int one_time;
    p->sent = 0;

    while (1)
    {
        one_time = send(p->fsd, data, sizeof(data), 0);
        p->sent += one_time;
    }

    return NULL;
}

void client (const char *host, const char *port, const char *time)
{
    int fsd;
    double elapsed;
    pthread_t child;
    struct addrinfo hints, *result, *p;
    struct parameter param;
    struct timespec connect_start, connect_end;


    if ((fsd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
        error("Error: client cannot create socket!");

    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = 0;
    hints.ai_protocol = 0;
    if (getaddrinfo(host, port, &hints, &result) != 0)
        error("Error: client cannot resolve host!");

    for (p = result; p != NULL; p = p->ai_next)
    {
        if (connect(fsd, p->ai_addr, p->ai_addrlen) != -1)
            break;
    }

    if (p == NULL)
        error("Error: client cannot connect to server!");
    memset(&data, 0, sizeof(data));
    memset(&buffer, 0, sizeof(buffer));

    param.fsd = fsd;
    clock_gettime(CLOCK_REALTIME, &connect_start);
    if (pthread_create(&child, NULL, &send_data, (void *)&param) != 0)
        error("Error: client cannot create send thread!");

    sleep((unsigned int) atoi(time));
    pthread_cancel(child);
    pthread_join(child, NULL);

    send(fsd, "FIN", 4, 0);

    recv(fsd, &buffer, 4, 0);
    if (memcmp(&buffer, "ACK", 3) != 0)
        error("Error: client cannot receive ACK");
    close(fsd);
    clock_gettime(CLOCK_REALTIME, &connect_end);

    elapsed = connect_end.tv_sec - connect_start.tv_sec +
              (connect_end.tv_nsec - connect_start.tv_nsec) / 1e9;
    printf("Sent=%d KB rate= %.3f Mbps\n", param.sent / 1000,
           param.sent / (elapsed * 1e6) * 8);
}