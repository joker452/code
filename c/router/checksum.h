#ifndef __CHECK__
#define __CHECK__
#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<string.h>
#include<netinet/ip.h>

uint16_t check_sum(uint16_t *iphd, uint8_t header_length);
uint16_t count_check_sum(struct ip* iphead);

#endif
