#pragma once

#include "mat-params.h"
#include "gmp-es-mpz.h"


__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_mul(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_mul_strided(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_addmul(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_addmul_strided(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_add(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_add_strided(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_sub(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_sub_strided(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

// vim: set sw=4 ts=4 noet:
