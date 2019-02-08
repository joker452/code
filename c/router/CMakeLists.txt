cmake_minimum_required(VERSION 3.12)
project(router)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(router
        arpfind.c
        arpfind.h
        checksum.c
        checksum.h
        lookuproute.c
        lookuproute.h
        Makefile
        recvroute.c
        recvroute.h
        sendetherip.c
        sendetherip.h error.c error.h
        t1.c)
