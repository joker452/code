#include "lookuproute.h"


int insert_route(unsigned long  ip4prefix,unsigned int prefixlen,
        char *ifname,unsigned int ifindex,unsigned long  nexthopaddr)
{
    // what is the next* in stuct nexthop for???
    struct route *next;
    if ((next = (struct route*) malloc(sizeof(struct route))) == NULL)
    {
        printf("malloc error at next!!\n");
        return -1;
    }
    memset(next, 0, sizeof(struct route));

    next->ip4prefix.s_addr = (in_addr_t) ip4prefix;
    next->prefixlen = prefixlen;
    if ((next->nexthop = (struct nexthop *) malloc(sizeof(struct nexthop))) == NULL)
    {
        printf("malloc error at nexthop!!\n");
        return -1;
    }
    memset(next->nexthop, 0, sizeof(struct nexthop));

    if ((next->nexthop->ifname = (char *) malloc((strlen(ifname) + 1) * sizeof(char))) == NULL)
    {
        printf("malloc error at ifname!!\n");
        return -1;
    }
    strcpy(next->nexthop->ifname, ifname);
    next->nexthop->ifindex = ifindex;
    next->nexthop->nexthopaddr.s_addr = (in_addr_t) nexthopaddr;

    next->next = route_table->next;
    route_table->next = next;

    return 1;

}

int lookup_route(struct in_addr dstaddr,
        struct nextaddr *nexthopinfo)
{
	unsigned int max_match = 0;
	struct route *next = route_table->next;

	while (next != NULL)
    {
	    in_addr_t net_address = next->ip4prefix.s_addr >> (32 - next->prefixlen);
	    in_addr_t and_result = dst.s_addr >> 32 - next->prefixlen;
	    if (and_result == net_address && next->prefixlen > max_match)
        {
            if ((nexthopinfo->ifname = (char *) malloc((strlen(next->nexthop->ifname) + 1) * sizeof(char))) == NULL)
            {
                printf("malloc error at ifname!!\n");
                return -1;
            }
            strcpy(nexthopinfo->ifname, ifname);
            nexthopinfo->ipv4addr = next->ip4prefix;
            nexthopinfo->prefixl = next->prefixlen;
	        max_match = next->prefixlen;
        }
        next = next->next;
    }

    return 1;
}

int delete_route(struct in_addr dstaddr, unsigned int prefixlen)
{
	struct route *this = route_table, * next;

	while (this->next != NULL)
    {
	    next = this->next;
	    if (next->ip4prefix.s_addr == dstaddr.s_addr && next->prefixlen == prefixlen)
        {
	        this->next = next->next;
	        // nexthop is a list, is it necessary to free the whole list?
	        free(next->nexthop);
	        free(next);
	        return 1;
        }
    }

    return -1;
}

