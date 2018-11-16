#include "checksum.h"

int check_sum(unsigned short *iphd,int len,unsigned short checksum)
{
	unsigned int header_checksum = 0, i;
	unsigned short mask = 0xffff;
	for (i = 0; i < len; ++i)
	{
		header_checksum += iphd[i];
	}

	while(header_checksum >> 16)
	{
		header_checksum = (header_checksum >> 16) + header_checksum & mask;
	}

	return (unsigned short) header_checksum == mask;
}
unsigned short count_check_sum(unsigned short *iphd)
{
    struct _iphdr * ip_head = (struct _iphdr *) iphd;
    int header_len = (ip_head->h_verlen & 0xf) * 2;
    ip_head->ttl = ip_head->ttl - 1;
    ip_head->checksum = 0;


    int i, header_checksum = 0;

    for (i = 0; i < header_len; ++i)
    {
        header_checksum += iphd[i];
    }

    while(header_checksum >> 16)
    {
        header_checksum = (header_checksum >> 16) + (header_checksum & 0xffff);
    }

    ip_head->checksum = ~(header_checksum & 0xffff);

    return ip_head->checksum;

}
