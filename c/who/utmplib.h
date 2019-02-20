//
// Created by deng on 19-2-20.
//
#ifndef WHO_UTMPLIB_H
#define WHO_UTMPLIB_H
#include        <stdio.h>
#include        <fcntl.h>
#include        <sys/types.h>
#include        <utmp.h>
int utmp_open( char *);
struct utmp *utmp_next();
int utmp_reload();
int utmp_close();
off_t utmp_seek(off_t, int);
#endif //WHO_UTMPLIB_H
