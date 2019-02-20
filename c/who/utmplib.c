/* utmplib.c  - functions to buffer reads from utmp file
 *
 *      functions are
 *              utmp_open( filename )   - open file
 *                      returns -1 on error
 *              utmp_next( )            - return pointer to next struct
 *                      returns NULL on eof
 *              utmp_close()            - close file
 *
 *      reads NRECS per read and then doles them out from the buffer */
#include "utmplib.h"
#define NRECS   16
#define NULLUT  ((struct utmp *)NULL)
#define UTSIZE  (sizeof(struct utmp))

static  char    utmpbuf[NRECS * UTSIZE];                /* storage      */
static  int     num_recs;                               /* num stored   */
static  int     cur_rec;                                /* next to go   */
static  int     fd_utmp = -1;                           /* read from    */

int utmp_open( char *filename )
{
}

struct utmp *utmp_next()
{
}

int utmp_reload()
/*
 *      read next bunch of records into buffer
 */
{
}

int utmp_close()
{

}
off_t utmp_seek(off_t record_offset, int base)
{

}