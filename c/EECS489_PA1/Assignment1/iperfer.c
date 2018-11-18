#include <getopt.h>
#include <ctype.h>
#include "iperfer.h"

void error(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    int opt, mode = -1, optind_before = optind;
    const char *host = NULL, *port = NULL, *time = NULL;
    opterr = 0;

    while ((opt = getopt(argc, argv, "sch:p:t:")) != -1)
    {
        if (optind == optind_before)
            error("Error: argument format incorrect!");
        switch (opt)
        {
            case 'c': mode = 0; break;
            case 's': mode = 1; break;
            case 'h': host = optarg; break;
            case 'p': port = optarg; break;
            case 't': time = optarg; break;
            case '?':
                if (optopt != 'h' && optopt != 'p' && optopt != 't')
                {
                    if (isprint(optopt))
                        fprintf(stderr, "Error: unknown option -%d!\n", optopt);
                    else
                        fprintf(stderr, "Error: unknown option -\\x%x!", optopt);
                    exit(EXIT_FAILURE);
                }
                else
                {
                    fprintf(stderr, "Error: missing arguments for option -%c\n", optopt);
                    error("Useage: iperfer [-s/-c] [-p port] [-h host] [-t time]");
                }
                break;
            default: error("Error: unknown error in iperfer!");
        }
        optind_before = optind;
    }

    if (mode == -1)
    {
        fprintf(stderr, "Error: no mode is specified!\n");
        error("Useage: iperfer [-s/-c] [-p port] [-h host] [-t time]");
    }
    else if (mode == 1)
    {
        /* server */
        if (argc == 4 && port != NULL)
            if (1024 <= atoi(port) && atoi(port) <= 65535)
                server(port);
            else
                error("Error: port number must be in the range [1024, 65535]");
        else
            error("Error: missing or additional arguments");
    }
    else
    {
        /* client */
        if (argc == 8 && host != NULL && port != NULL && time != NULL)
            if (1024 <= atoi(port) && atoi(port) <= 65535)
                client(host, port, time);
            else
                error("Error: port number must be in the range [1024, 65535]");
        else
            error("Error: missing or additional arguments");
    }

    return 0;
}