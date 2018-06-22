CC := gcc
CFLAGS := -Wall -Wextra -std=c11 -pthread -O3 -flto -march=native -mtune=native -g
LDFLAGS := -lm -pthread -flto

TARGETS := main

OBJECTS := $(patsubst %.c,%.o,$(wildcard *.c))


.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f $(TARGETS) *.o


main: $(OBJECTS) libgmp_asm_tuned.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(wildcard *.h)
	$(CC) $(CFLAGS) -c -o $@ $<
