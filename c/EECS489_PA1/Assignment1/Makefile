iPerfer:client.o server.o iperfer.o iperfer.h
	gcc -Wall -Og -lrt -pthread -o iPerfer client.o server.o iperfer.o iperfer.h
client.o:client.c iperfer.h
	gcc -Wall -Og -c -o client.o client.c
server.o:server.c iperfer.h
	gcc -Wall -Og -c -o server.o server.c
iperfer.o:iperfer.c iperfer.h
	gcc -Wall -Og -c -o iperfer.o iperfer.c
clean:
	rm -f *.o
	rm -f iPerfer
