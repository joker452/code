//
// Created by deng on 19-5-2.
//

#ifndef SERVER_UTIL_H
#define SERVER_UTIL_H
#define MAX_EVENTS 1024
#include <sys/epoll.h>          /* epoll */
int tcp_epoll_create(int flags);
void tcp_epoll_add(int epfd, int fd, struct epoll_event *event);
void tcp_epoll_mod(int epfd, int fd, struct epoll_event *event);
void tcp_epoll_del(int epfd, int fd, struct epoll_event *event);
int tcp_epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);
#endif //SERVER_EPOLL_H
