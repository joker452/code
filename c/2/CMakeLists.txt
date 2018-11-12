cmake_minimum_required(VERSION 3.12)
project(2 C)

set(CMAKE_C_STANDARD 11)

include_directories(forward框架)

add_executable(2
        forward框架/arpfind.c
        forward框架/arpfind.h
        forward框架/checksum.c
        forward框架/checksum.h
        forward框架/lookuproute.c
        forward框架/lookuproute.h
        forward框架/main.c
        forward框架/recvroute.c
        forward框架/recvroute.h
        forward框架/sendetherip.c
        forward框架/sendetherip.h)
