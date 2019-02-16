#include "arpfind.h"
#include <net/if.h>
#include "error.h"
int arpGet(struct arpmac *srcmac, char *ifname, char *ipStr)
{
    struct ifreq out_if;
    struct arpreq arp;
    struct sockaddr_in *target_addr;
    int arpsfd;
    if (ifname != NULL) {
        memset(&out_if, 0, sizeof(out_if));
        snprintf(out_if.ifr_ifrn.ifrn_name, sizeof(ifname), "%s", ifname);
        if ((arpsfd = socket(AF_INET, SOCK_RAW, IPPROTO_RAW)) < 0) {
            perror("Error in function socket");
            exit(EXIT_FAILURE);
        }
        else {
            if (ioctl(arpsfd, SIOCGIFHWADDR, &out_if) < 0) {
                perror("Error in function ioctl");
                exit(EXIT_FAILURE);
            }

            if ((srcmac->mac = (unsigned char*) malloc(sizeof(out_if.ifr_ifru.ifru_hwaddr))) == NULL)
                error("Error in function malloc: mac address allocated fail");
            else
                if ((srcmac->indoex = if_nametoindex(ifname)) == 0) {
                    perror("Error in function if_nametoindex");
                    exit(EXIT_FAILURE);
                }
                else {
                    memset(&arp, 0, sizeof(arp));
                    target_addr = (struct sockaddr_in *) &arp.arp_pa;
                    target_addr->sin_family = AF_INET;
                    target_addr->sin_addr.s_addr = inet_addr(ipStr);
                    snprintf(arp.arp_dev, sizeof(ifname), "%s", ifname);
                    if (ioctl(arpsfd, SIOCGARP, &arp) < 0) {
                        perror("Error in function ioctl");
                        exit(EXIT_FAILURE);
                    }

                }

        }
    }
    else {
        error("Error in function arpGet: ifname is null!");
    }

    return 0;  
}  
                                                                                                        
                                                                                                          
                                                                                                            
                                                                                                              
