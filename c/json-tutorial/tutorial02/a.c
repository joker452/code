#include "leptjson.h"
#include <stdio.h>
typedef struct {
    char* s;
    size_t len;
    double n;
    lept_type type;
}lept_value;

int main()
{
	printf("%d", sizeof(lept_value));
}
