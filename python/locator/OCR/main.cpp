#include "util.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc == 2) {
        string file = argv[1];
        processManuscripts(file);
    }
    else {
        cout << "Usage: [FILE]" << endl;

    }

    return 0;
}



