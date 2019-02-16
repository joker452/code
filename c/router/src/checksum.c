#include "checksum.h"

uint16_t check_sum(uint16_t *iphd, uint8_t header_length)
{
   uint8_t i = 0, end = header_length >> 1;
   uint32_t sum = 0;

   while (i < end)
   {
       sum = sum + *(iphd + i);
       ++i;
   }

   while (sum >> 16)
   {
       sum = (sum >> 16) + (sum & 0xffff);
   }

    return (uint16_t) ~sum;
}

uint16_t count_check_sum(struct ip* iphead)
{
    --iphead->ip_ttl;
    iphead->ip_sum = 0;
    return check_sum((uint16_t *) iphead, (uint8_t) iphead->ip_hl << 2);
}
