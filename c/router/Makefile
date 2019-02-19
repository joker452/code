vpath %.h ./include
vpath %.c ./src
CFLAGS = -Og -Wall -Wextra -std=c99
CC = gcc
OBJECTS = $(wildcard *.o)

%.o: %c %.h
	$(CC) -o $@ $(CFLAGS) -c $<
main : $(OBJECTS)
	echo $(OBJECTS)
	$(CC) -o $@  $^ $(CFLAGS) -lpthread
.PHONY: clean all
SUBDIR = $(shell ls ./ -R | grep /)
SUBDIRS  = $(subst :,/,$(SUBDIR))
TEMP = $(foreach dir, $(SUBDIRS),$(wildcard $(dir)*.o))
clean:
	rm  -rf $(TEMP)
