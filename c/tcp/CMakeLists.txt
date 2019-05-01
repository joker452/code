cmake_minimum_required(VERSION 3.14)
project(tcp C)
file(GLOB SOURCES "src/*.c")
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-Og -Wall -Wextra -pthread -DNDEBUG")
set(CMAKE_C_STANDARD 99)
add_executable(server ${SOURCES})

