#pragma once

#include <stdint.h>
#include "gmp-es-mpz.h"


int64_t gettimestamp(void);

// Returns m such that m divides n but b does not divide m
unsigned int remove_factors(unsigned int n, unsigned int b);

// Returns whether n is a power of b
bool is_power_of(unsigned int n, unsigned int b);


inline const struct gpu___mpz_t* submatrix_ltwh(
		const struct gpu___mpz_t *M, gpu_mp_size_t rowstride, gpu_mp_size_t ltw, gpu_mp_size_t lth, gpu_mp_size_t ix, gpu_mp_size_t iy) {
	return M + iy * (lth * rowstride) + ix * ltw;
}

inline struct gpu___mpz_t* submatrix_ltwh(
		struct gpu___mpz_t *M, gpu_mp_size_t rowstride, gpu_mp_size_t ltw, gpu_mp_size_t lth, gpu_mp_size_t ix, gpu_mp_size_t iy) {
	return M + iy * (lth * rowstride) + ix * ltw;
}

inline const struct gpu___mpz_t* submatrix(
		gpu_mp_size_t W, gpu_mp_size_t H, const struct gpu___mpz_t *M, gpu_mp_size_t rowstride, gpu_mp_size_t ix, gpu_mp_size_t iy) {
	return submatrix_ltwh(M, rowstride, W / 2, H / 2, ix, iy);
}

inline struct gpu___mpz_t* submatrix(
		gpu_mp_size_t W, gpu_mp_size_t H, struct gpu___mpz_t *M, gpu_mp_size_t rowstride, gpu_mp_size_t ix, gpu_mp_size_t iy) {
	return submatrix_ltwh(M, rowstride, W / 2, H / 2, ix, iy);
}


// vim: set sw=4 ts=4 noet:
