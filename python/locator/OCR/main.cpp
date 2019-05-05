#include "util.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc == 2) {
        string file = argv[1];
        timespec start, end;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        processManuscripts(file);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        timespec dif = diff(start, end);
        cout << dif.tv_sec << "s plus " << dif.tv_nsec << "ns" << endl;
    }
    else {
        cout << "Usage: [FILE]" << endl;

    }

    return 0;
}



