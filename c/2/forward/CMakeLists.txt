cmake_minimum_required(VERSION 3.12)
project(forward C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(forward
    arpfind.c
    arpfind.h
    checksum.c
    checksum.h
    lookuproute.c
    lookuproute.h
    main.c
    recvroute.c
    recvroute.h
    sendetherip.c
    sendetherip.h)
