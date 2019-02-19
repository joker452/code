cmake_minimum_required(VERSION 3.1)
project(router)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
# the way to link here only works with CMake 3.1.0+
find_package(Threads REQUIRED)

add_executable(router
        src/arpfind.c
        src/checksum.c
        src/lookuproute.c
        src/recvroute.c
        src/sendetherip.c
        src/error.c
        test/arp_test.c)
target_include_directories(router
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(router Threads::Threads)
