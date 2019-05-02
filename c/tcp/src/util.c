#include <stdlib.h>
#include "util.h"
#include "error.h"


struct epoll_event *events;

int tcp_epoll_create(int flags)
{
    int fd;

    if ((fd = epoll_create1(flags)) == -1)
        unix_error("error in tcp_epoll_create: cannot create epoll instance");

    events = (struct epoll_event *)malloc(sizeof(struct epoll_event) * MAX_EVENTS);
    if (events == NULL)
        app_error("error in tcp_epoll_create: cannot allocate memory");
    return fd;
}

void tcp_epoll_add(int epfd, int fd, struct epoll_event *event)
{
    if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, event) == - 1)
        unix_error("error in tcp_epoll_add");

}

int tcp_epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout)
{
    int n;
    if ((n = epoll_wait(epfd, events, maxevents, timeout)) == -1)
        unix_error("error in tcp_epoll_wait");
    return n;
}

