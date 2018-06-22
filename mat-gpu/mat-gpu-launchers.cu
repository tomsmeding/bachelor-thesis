#include "mat-gpu-launchers.h"
#include "mat-core.h"


__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_mul(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	gpu_mp_size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	gpu_mp_size_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= N || y >= K) return;

	mat_mul_1(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, x, y);
}

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_mul_strided(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	gpu_mp_size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	gpu_mp_size_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= N || y >= K) return;

	mat_mul_1_strided(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, x, y);
}

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_addmul(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	gpu_mp_size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	gpu_mp_size_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= N || y >= K) return;

	mat_addmul_1(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, x, y);
}

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_addmul_strided(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	gpu_mp_size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	gpu_mp_size_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= N || y >= K) return;

	mat_addmul_1_strided(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, x, y);
}

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_add(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	gpu_mp_size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	gpu_mp_size_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= W || y >= H) return;

	gpu_mpz_add(&C[Crowstride * y + x], &A[Arowstride * y + x], &B[Browstride * y + x]);
}

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_add_strided(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	const gpu_mp_size_t Alimbstride = INTERLEAVE_BLOCK_SIZE;
	const gpu_mp_size_t Blimbstride = INTERLEAVE_BLOCK_SIZE;
	const gpu_mp_size_t Climbstride = INTERLEAVE_BLOCK_SIZE;

	gpu_mp_size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	gpu_mp_size_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= W || y >= H) return;

	gpu_mp_limb_t *Cp = (gpu_mp_limb_t*)&C[Crowstride * y + x / Climbstride * Climbstride] + x % Climbstride;
	const gpu_mp_limb_t *Ap = (gpu_mp_limb_t*)&A[Arowstride * y + x / Alimbstride * Alimbstride] + x % Alimbstride;
	const gpu_mp_limb_t *Bp = (gpu_mp_limb_t*)&B[Browstride * y + x / Blimbstride * Blimbstride] + x % Blimbstride;

	struct gpu___mpz_t *Czp = (struct gpu___mpz_t*)Cp;
	const struct gpu___mpz_t *Azp = (struct gpu___mpz_t*)Ap;
	const struct gpu___mpz_t *Bzp = (struct gpu___mpz_t*)Bp;

	gpu_mpz_add_strided(Czp, Climbstride, Azp, Alimbstride, Bzp, Blimbstride);
}

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_sub(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	gpu_mp_size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	gpu_mp_size_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= W || y >= H) return;

	gpu_mpz_sub(&C[Crowstride * y + x], &A[Arowstride * y + x], &B[Browstride * y + x]);
}

__launch_bounds__(GPU_BLK_W * GPU_BLK_H)
__global__ void launcher_sub_strided(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	const gpu_mp_size_t Alimbstride = INTERLEAVE_BLOCK_SIZE;
	const gpu_mp_size_t Blimbstride = INTERLEAVE_BLOCK_SIZE;
	const gpu_mp_size_t Climbstride = INTERLEAVE_BLOCK_SIZE;

	gpu_mp_size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	gpu_mp_size_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= W || y >= H) return;

	gpu_mp_limb_t *Cp = (gpu_mp_limb_t*)&C[Crowstride * y + x / Climbstride * Climbstride] + x % Climbstride;
	const gpu_mp_limb_t *Ap = (gpu_mp_limb_t*)&A[Arowstride * y + x / Alimbstride * Alimbstride] + x % Alimbstride;
	const gpu_mp_limb_t *Bp = (gpu_mp_limb_t*)&B[Browstride * y + x / Blimbstride * Blimbstride] + x % Blimbstride;

	struct gpu___mpz_t *Czp = (struct gpu___mpz_t*)Cp;
	const struct gpu___mpz_t *Azp = (struct gpu___mpz_t*)Ap;
	const struct gpu___mpz_t *Bzp = (struct gpu___mpz_t*)Bp;

	gpu_mpz_sub_strided(Czp, Climbstride, Azp, Alimbstride, Bzp, Blimbstride);
}

// vim: set sw=4 ts=4 noet:
