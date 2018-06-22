#pragma once

#include <stdio.h>
#include "gmp.h"


typedef mpz_t *mat_t;
typedef const mpz_t *mat_src_t;

// A rowstride of n indicates that the rows in the backing array are n mpz_t's wide.

mat_t mat_init(unsigned int W, unsigned int H);
void mat_free(unsigned int W, unsigned int H, mat_t A);
void mat_random(unsigned int W, unsigned int H, mat_t A, int nbits);
void mat_random_mpz(mpz_t z, int nbits);

// If not used, generator is seeded with a high-precision timestamp
void mat_seed_generator(unsigned long int seed);

void mat_add(unsigned int W, unsigned int H, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs);
void mat_sub(unsigned int W, unsigned int H, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs);

void mat_mul(unsigned int K, unsigned int M, unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs);
void mat_addmul(unsigned int K, unsigned int M, unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs);
void mat_mul_strassen(unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, unsigned int nlevels);
void mat_mul_winograd(unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, unsigned int nlevels);
void mat_addmul_winograd(unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, unsigned int nlevels);

void mat_read(unsigned int W, unsigned int H, FILE *f, mat_t A);
void mat_write(unsigned int W, unsigned int H, FILE *f, mat_src_t A, mp_size_t Arowstride);
void mat_write_oneline(unsigned int W, unsigned int H, FILE *f, mat_src_t A, mp_size_t Arowstride);

// vim: set sw=4 ts=4 noet:
