#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "iperfer.h"

#define LISTEN_BACKLOG 5


void server(const char *port)
{
    int sfd, cfd;
    int received = 0, one_time = 0;
    double elapsed;
    uint16_t port_num = (uint16_t) atoi(port);
    socklen_t cli_len;
    char buffer[MAX_BUF];
    struct sockaddr_in serv_addr, cli_addr;
    memset(&buffer, 0, sizeof(buffer));

    if ((sfd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
        error("Error: server cannot create socket!");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port_num);
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    
    if (bind(sfd, (struct sockaddr *) &serv_addr, 
            sizeof(struct sockaddr_in)) == -1)
        error("Error: server cannot bind socket!");
    
    if (listen(sfd, LISTEN_BACKLOG) == -1)
        error("Error: server cannot listen!");

    struct timespec connect_start, connect_end;

    memset(&cli_addr, 0, sizeof(cli_addr));
    cli_len = sizeof(cli_addr);
    cfd = accept(sfd, (struct sockaddr *) &cli_addr, &cli_len);

    clock_gettime(CLOCK_REALTIME, &connect_start);
    while (1)
    {
        if ((one_time = recv(cfd, &buffer, MAX_BUF, 0)) == -1)
            error("Error: server cannot receive data!");

        received += one_time;
        if (!memcmp(&buffer, "FIN", 4)) break;
    }
    send(cfd, "ACK", 4, 0);
    close(cfd);
    clock_gettime(CLOCK_REALTIME, &connect_end);

    elapsed = connect_end.tv_sec - connect_start.tv_sec +
            (connect_end.tv_nsec - connect_start.tv_nsec) / 1e9;

    printf("Recevied=%d KB rate= %.3f Mbps\n", received / 1000,
            received / (elapsed * 1e6) * 8);
}

