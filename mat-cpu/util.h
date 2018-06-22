#pragma once

#include <stdbool.h>
#include <stdint.h>


int64_t gettimestamp(void);

inline bool is_power_of_2(unsigned int n) {
	return (n & (n-1)) == 0;
}

// Returns m such that m divides n but b does not divide m
unsigned int remove_factors(unsigned int n, unsigned int b);

// Returns whether n is a power of b
bool is_power_of(unsigned int n, unsigned int b);

// Returns highest k such that 2^k <= n; undefined for n == 0
unsigned int round_down_power2(unsigned int n);

// vim: set sw=4 ts=4 noet:
