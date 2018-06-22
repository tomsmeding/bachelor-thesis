#pragma once

#include <stdio.h>
#include <stdlib.h>


#define CUSUC(expr__) \
	do { \
		cudaError_t e__ = (expr__); \
		if (e__ != cudaSuccess) { \
			fprintf(stderr, "On %s:%d  %s\n", __FILE__, __LINE__, #expr__); \
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e__)); \
			abort(); \
		} \
	} while(0)

// vim: set sw=4 ts=4 noet:
