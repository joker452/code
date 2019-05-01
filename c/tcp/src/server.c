#include <stdio.h>
#include <unistd.h>         /* getopt */
#include <stdlib.h>
#include <string.h>         /* memset */
#include <errno.h>          /* errno */
#include <sys/types.h>      /* getaddrinfo */
#include <sys/socket.h>     /* getaddrinfo */
#include <netdb.h>          /* getaddrinfo */
#include <sys/socket.h>     /* setsockopt */

#define BACKLOG 1024
#include "error.h"
static void usage()
{
    fprintf(stderr, "usage: <port>");
}

int main(int argc, char* argv[])
{
    int opt, p, sfd, addr_res, optval = 1;
    char port[10], *endptr = NULL;
    while ((opt = getopt(argc, argv, "p:")) != -1)
        switch (opt)
        {
            case 'p':
                p = strtol(optarg, &endptr, 10);
                if (endptr == optarg)
                    app_error("error in main: no digits found for port");
                else if (errno != 0)
                    unix_error("error in main: parse port failure");
                else if (p < 1025 || p > 65535)
                    app_error("error in main: port should be a number between [1025, 65535]");
                snprintf(port, sizeof(port), "%s", optarg);
                break;
            case '?':
                usage();
                break;
        }

    struct addrinfo hints, *server_addr, *addr_ptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_protocol = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addr_res = getaddrinfo(NULL, port, &hints, &server_addr);
    if (addr_res != 0)
        gai_error(errno, "error in main: cannot get server addrness info");
    for (addr_ptr = server_addr; addr_ptr != NULL; addr_ptr = addr_ptr->ai_next)
    {
        if ((sfd = socket(addr_ptr->ai_family, addr_ptr->ai_socktype, addr_ptr->ai_protocol)) == -1)
            continue;
        if (bind(sfd, addr_ptr->ai_addr, addr_ptr->ai_addrlen) == 0)
            break;
        close(sfd);
    }

    if (addr_ptr == NULL)
        app_error("error in main: cannot find available address for bind");

    if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof (optval)) == -1)
        unix_error("error in main: cannot set socket option");

    return 0;
}