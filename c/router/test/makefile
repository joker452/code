main: arpfind.o arp_test.o error.o
	gcc -Wall -Wextra -std=c99 -Og -o main $^
%.o: %.c %.h
	gcc -Wall -Wextra -std=c99 -Og -o $@ -c $<
.PHONY: clean
clean:
	rm -f *.o main
