#include <stddef.h>
#include <sys/time.h>
#include "util.h"


int64_t gettimestamp(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000L + tv.tv_usec;
}

unsigned int remove_factors(unsigned int n, unsigned int b) {
	while (true) {
		unsigned int d = n / b, r = n % b;
		if (r == 0) n = d;
		else return n;
	}
}

bool is_power_of(unsigned int n, unsigned int b) {
	return remove_factors(n, b) == 1;
}

// vim: set sw=4 ts=4 noet:
