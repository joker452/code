#include <stdio.h>
#include <unistd.h>         /* getopt, fcntl*/
#include <fcntl.h>          /* fcntl */
#include <stdlib.h>
#include <string.h>         /* memset */
#include <errno.h>          /* errno */
#include <sys/types.h>      /* getaddrinfo */
#include <sys/socket.h>     /* getaddrinfo */
#include <netdb.h>          /* getaddrinfo */
#include <sys/socket.h>     /* setsockopt */

#define BACKLOG 1024
#include "error.h"
#include "util.h"

extern struct epoll_event *events;

static void usage()
{
    fprintf(stderr, "usage: <port>");
}

void set_nonblocking(int fd)
{
    int flags;
    if ((flags = fcntl(fd, F_GETFL, 0)) == - 1)
        unix_error("error in set_nonblocking: cannot get file status");
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1)
        unix_error("error in set_nonblocking: cannot set non blocking socket");

}

int main(int argc, char* argv[])
{
    int opt, p, epfd, n, i, sfd, fd, addr_res, optval = 1;
    char port[10], *endptr = NULL;
    struct addrinfo hints, *server_addr, *addr_ptr;
    struct sockaddr_storage client_addr;
    struct epoll_event event;
    socklen_t  addr_size = sizeof(client_addr);

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

    printf("%s", port);
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addr_res = getaddrinfo(NULL, port, &hints, &server_addr);
    if (addr_res != 0)
        gai_error(errno, "error in main: cannot get server addrness info");
    for (addr_ptr = server_addr; addr_ptr != NULL; addr_ptr = addr_ptr->ai_next)
    {
        if ((sfd = socket(addr_ptr->ai_family, addr_ptr->ai_socktype, addr_ptr->ai_protocol)) == -1)
            continue;

        if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof (optval)) == -1)
            unix_error("error in main: cannot set socket option");

        if (bind(sfd, addr_ptr->ai_addr, addr_ptr->ai_addrlen) == 0)
            break;
        /* create socket but cannot bind */
        close(sfd);
    }

    if (addr_ptr == NULL)
        app_error("error in main: cannot find available address for bind");

    freeaddrinfo(server_addr);
    if (listen(sfd, BACKLOG) == - 1)
        unix_error("error in main: cannot listen");

    set_nonblocking(sfd);
    epfd = tcp_epoll_create(0);
    event.events = EPOLLIN | EPOLLET;
    event.data.fd = sfd;
    tcp_epoll_add(epfd, sfd, &event);

    while (1)
    {
        n = tcp_epoll_wait(epfd, events, MAX_EVENTS, -1);
        for (i = 0; i < n; ++i)
        {
            fd = events[i].data.fd;
            if ((events[i].events & EPOLLERR) ||
                (events[i].events & EPOLLHUP) ||
                (!events[i].events & EPOLLIN))
            {
                close(fd);
                app_error("error in main: error occur on epoll fd");
            }
            else
            {
                if (fd == sfd)
                {
                    int infd;
                    while (1)
                    {
                        if ((infd = accept(sfd, (struct sockaddr *) &client_addr, &addr_size)) == -1)
                        {
                            if (errno == EAGAIN || errno == EWOULDBLOCK)
                                break;
                            else
                                unix_error("error in main: accept error");
                        }

                        set_nonblocking(infd);
                        printf("new connection from %d", infd);
                        event.data.fd = infd;
                        event.events = EPOLLIN | EPOLLET;
                        tcp_epoll_add(epfd, infd, &event);
                    }
                }
                else
                {
                    int count, done = 0;
                    char buf[512];
                    while (1)
                    {

                        count = read(fd, buf, sizeof buf);
                        if (count == -1)
                        {
                            /* If errno == EAGAIN, that means we have read all
                               data. So go back to the main loop. */
                            if (errno != EAGAIN)
                            {
                                app_error("error in main: cannot read");
                            }
                            break;
                        }
                        else if (count == 0)
                        {
                            /* End of file. The remote has closed the
                               connection. */
                            done = 1;
                            break;
                        }

                        /* Write the buffer to standard output */
                        printf("data from %d", fd);
                        if (write(1, buf, count) == -1)
                        {
                            unix_error("error in main: cannot write");
                        }
                    }
                    if (done)
                    {
                        printf ("Closed connection on descriptor %d\n",
                                events[i].data.fd);
                        close (events[i].data.fd);
                    }

                }
            }
        }
    }

    free(events);
    close(sfd);
    return 0;
}