/* who3.c - who with buffered reads
 *	  - surpresses empty records
 *	  - formats time nicely
 *	  - buffers input (using utmplib)
 */
#include <stdlib.h> /* exit */
#include <stdio.h> /* perror */
#include <sys/types.h>
#include <utmp.h>
#include <unistd.h> /* read */
#include <fcntl.h>
#include <time.h>
#include "utmplib.h"
#define	SHOWHOST
void error(const char *message);
void show_info(struct utmp *);
void showtime(time_t);

int main(int argc, char *argv[])
{
    struct utmp* log_record;
    if (utmp_open(UTMP_FILE) < 0)
        error("error in function utmp_open");
    else
    {
        while ((log_record = utmp_next()) != NULL)
            show_info(log_record);
    }
    utmp_close();
    return 0;

}

void error(const char* message) {
    perror(message);
    exit(EXIT_FAILURE);
}
/*
 *	show info()
 *			displays the contents of the utmp struct
 *			in human readable form
 *			* displays nothing if record has no user name
 */
void show_info( struct utmp *utbufp)
{

}

void showtime( time_t timeval )
/*
 *	displays time in a format fit for human consumption
 *	uses ctime to build a string then picks parts out of it
 *      Note: %12.12s prints a string 12 chars wide and LIMITS
 *      it to 12chars.
 */
{

}