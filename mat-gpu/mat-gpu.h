#pragma once

#include "gmp-es-mpz.h"


void add_matrices_gpu(
		gpu_mp_size_t W, gpu_mp_size_t H,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void sub_matrices_gpu(
		gpu_mp_size_t W, gpu_mp_size_t H,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void add_matrices_strided_gpu(
		gpu_mp_size_t W, gpu_mp_size_t H,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void sub_matrices_strided_gpu(
		gpu_mp_size_t W, gpu_mp_size_t H,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);


void naive_smallrect_gpu(
		gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);


void naive_recursive_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void naive_recursive_strided_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void strassen_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void strassen_strided_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_addmul_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_strided_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_addmul_strided_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

// vim: set sw=4 ts=4 noet:
