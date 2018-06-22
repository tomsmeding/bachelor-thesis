#pragma once

#include "gmp-es-mpz.h"


void naive_recursive_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void naive_recursive_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void strassen_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void strassen_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_addmul_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_addmul_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void squarify_mul_pregpu(
		gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride,
		void (*squarefunc)(
				gpu_mp_size_t N,
				struct gpu___mpz_t *C, gpu_mp_size_t Crs,
				const struct gpu___mpz_t *A, gpu_mp_size_t Ars,
				const struct gpu___mpz_t *B, gpu_mp_size_t Brs));

// vim: set sw=4 ts=4 noet:
