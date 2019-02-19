//
// Created by deng on 2/19/19.
//
#include "../include/arpfind.h"
#include "../include/error.h"
#include "stdlib.h"
#define DEBUG 1
int main(int argc, char *argv[]) {
    struct arpmac *srcmac = (struct arpmac *) malloc(sizeof(struct arpmac));
    if (srcmac == NULL) {
        error("error in malloc in main");
    }
    arpGet(srcmac, "eth1", "101.6.69.194");
}
