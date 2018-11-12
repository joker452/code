#include <stdio.h>

void main() {

		int num = 4;
		switch (num) {
		case 5:
			printf("%d\n", 5);
		default:
			printf("%d\n", -1);
		case 3:
			printf("%d\n", 3);
			return;
		case 1:
		case 4:
			printf("%d\n", 4);
			break;
 		}
	}
