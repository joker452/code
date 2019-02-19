#include "checksum.h"
int main() {
    uint16_t hd[10] = {0x4500, 0x73, 0, 0x4000, 0x4011, 0xb861, 0xc0a8,
                       1, 0xc0a8, 0xc7};
    uint16_t length = 20;
    int a = check_sum(hd, 20);
    struct ip iphead;
    iphead.ip_hl = 0;
    iphead.ip_v = 0;
    iphead.ip_tos = 0x45;
    iphead.ip_len = 0x73;
    iphead.ip_id = 0;
    iphead.ip_off = 0x4000;
    iphead.ip_ttl = 0x11;
    iphead.ip_p = 0x40;
    iphead.ip_sum = 0xb861;
    iphead.ip_src.s_addr = 0x0001c0a8;
    iphead.ip_dst.s_addr = 0x00c7c0a8;
    uint16_t b = count_check_sum(&iphead);
    return 0;
}