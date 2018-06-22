#include <stdio.h>
#include <assert.h>
#include "mat-defines.h"
#include "mat-gpu.h"
#include "mat-gpu-launchers.h"
#include "cusuc.h"
#include "util.h"

void add_matrices_gpu(
		gpu_mp_size_t W, gpu_mp_size_t H,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "add_matrices_gpu(%d x %d)\n", W, H);

	launcher_add
		<<<dim3((W + GPU_BLK_W - 1) / GPU_BLK_W, (H + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
		(W, H, C, Crowstride, A, Arowstride, B, Browstride);
	CUSUC(cudaDeviceSynchronize());
}

void sub_matrices_gpu(
		gpu_mp_size_t W, gpu_mp_size_t H,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "sub_matrices_gpu(%d x %d)\n", W, H);

	launcher_sub
		<<<dim3((W + GPU_BLK_W - 1) / GPU_BLK_W, (H + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
		(W, H, C, Crowstride, A, Arowstride, B, Browstride);
	CUSUC(cudaDeviceSynchronize());
}

void add_matrices_strided_gpu(
		gpu_mp_size_t W, gpu_mp_size_t H,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "add_matrices_strided_gpu(%d x %d)\n", W, H);

	launcher_add_strided
		<<<dim3((W + GPU_BLK_W - 1) / GPU_BLK_W, (H + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
		(W, H, C, Crowstride, A, Arowstride, B, Browstride);
	CUSUC(cudaDeviceSynchronize());
}

void sub_matrices_strided_gpu(
		gpu_mp_size_t W, gpu_mp_size_t H,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "sub_matrices_strided_gpu(%d x %d)\n", W, H);

	launcher_sub_strided
		<<<dim3((W + GPU_BLK_W - 1) / GPU_BLK_W, (H + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
		(W, H, C, Crowstride, A, Arowstride, B, Browstride);
	CUSUC(cudaDeviceSynchronize());
}

void naive_smallrect_gpu(
		gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "naive_smallrect_gpu(%d x %d x %d)\n", K, M, N);

	launcher_mul
		<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (K + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
		(K, M, N, C, Crowstride, A, Arowstride, B, Browstride);
	CUSUC(cudaDeviceSynchronize());
}

void naive_recursive_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "naive_recursive_gpu(%d)\n", N);

	if (N <= BASE_SIZE) {
		launcher_mul
			<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (N + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
			(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		CUSUC(cudaDeviceSynchronize());
		return;
	}

	assert(N % 2 == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaMalloc(&T, (N / 2) * (N / 2) * sizeof(struct gpu___mpz_t)));

	const gpu_mp_size_t Trowstride = N / 2;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		naive_recursive_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	NAIVE_RECURSIVE_SEQUENCE;

#undef ADD
#undef MUL

	CUSUC(cudaFree(T));
}

void naive_recursive_strided_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "naive_recursive_strided_gpu(%d)\n", N);

	assert(INTERLEAVE_BLOCK_SIZE <= Crowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Arowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Browstride);

	if (N <= BASE_SIZE) {
		launcher_mul_strided
			<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (N + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
			(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		CUSUC(cudaDeviceSynchronize());
		return;
	}

	assert(N % 2 == 0);
	assert(N / 2 % INTERLEAVE_BLOCK_SIZE == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaMalloc(&T, (N / 2) * (N / 2) * sizeof(struct gpu___mpz_t)));

	const gpu_mp_size_t Trowstride = N / 2;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_strided_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		naive_recursive_strided_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	NAIVE_RECURSIVE_SEQUENCE;

#undef ADD
#undef MUL

	CUSUC(cudaFree(T));
}

void strassen_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "strassen_gpu(%d)\n", N);

	if (N <= BASE_SIZE) {
		// printf("#### %d %d %d\n", N, GPU_BLK_W, GPU_BLK_H);
		// launcher_mul
		//     <<<dim3(1, 1), dim3(1, 1)>>>
		//     (N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		launcher_mul
			<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (N + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
			(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		CUSUC(cudaDeviceSynchronize());
		return;
	}

	assert(N % 2 == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaMalloc(&T, N * N * sizeof(struct gpu___mpz_t)));

	const gpu_mp_size_t Trowstride = N;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);
	SPLIT_MATRIX_NUM(p, N, N, T, Trowstride, 1, 2, 3, 4);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define SUB(c_, cr_, a_, ar_, b_, br_) \
		sub_matrices_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		strassen_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	STRASSEN_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL

	CUSUC(cudaFree(T));
}

void strassen_strided_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "strassen_strided_gpu(%d)\n", N);

	assert(INTERLEAVE_BLOCK_SIZE <= Crowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Arowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Browstride);

	if (N <= BASE_SIZE) {
		launcher_mul_strided
			<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (N + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
			(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		CUSUC(cudaDeviceSynchronize());
		return;
	}

	assert(N % 2 == 0);
	assert(N / 2 % INTERLEAVE_BLOCK_SIZE == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaMalloc(&T, N * N * sizeof(struct gpu___mpz_t)));

	const gpu_mp_size_t Trowstride = N;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);
	SPLIT_MATRIX_NUM(p, N, N, T, Trowstride, 1, 2, 3, 4);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_strided_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define SUB(c_, cr_, a_, ar_, b_, br_) \
		sub_matrices_strided_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		strassen_strided_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	STRASSEN_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL

	CUSUC(cudaFree(T));
}

void winograd_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "winograd_gpu(%d)\n", N);

	if (N <= BASE_SIZE) {
		launcher_mul
			<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (N + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
			(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		CUSUC(cudaDeviceSynchronize());
		return;
	}

	assert(N % 2 == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaMalloc(&T, N * N * sizeof(struct gpu___mpz_t)));

	const gpu_mp_size_t Trowstride = N;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);
	SPLIT_MATRIX_NUM(p, N, N, T, Trowstride, 1, 2, 3, 4);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define SUB(c_, cr_, a_, ar_, b_, br_) \
		sub_matrices_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		winograd_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define AML(c_, cr_, a_, ar_, b_, br_) \
		winograd_addmul_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	WINOGRAD_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL
#undef AML

	CUSUC(cudaFree(T));
}

void winograd_addmul_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "winograd_addmul_gpu(%d)\n", N);

	if (N <= BASE_SIZE) {
		launcher_addmul
			<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (N + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
			(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		CUSUC(cudaDeviceSynchronize());
		return;
	}

	assert(N % 2 == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaMalloc(&T, N * N * sizeof(struct gpu___mpz_t)));

	const gpu_mp_size_t Trowstride = N;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);
	SPLIT_MATRIX_NUM(p, N, N, T, Trowstride, 1, 2, 3, 4);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define SUB(c_, cr_, a_, ar_, b_, br_) \
		sub_matrices_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		winograd_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define AML(c_, cr_, a_, ar_, b_, br_) \
		winograd_addmul_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	WINOGRAD_ADDMUL_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL
#undef AML

	CUSUC(cudaFree(T));
}

void winograd_strided_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "winograd_strided_gpu(%d)\n", N);

	assert(INTERLEAVE_BLOCK_SIZE <= Crowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Arowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Browstride);

	if (N <= BASE_SIZE) {
		launcher_mul_strided
			<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (N + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
			(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		CUSUC(cudaDeviceSynchronize());
		return;
	}

	assert(N % 2 == 0);
	assert(N / 2 % INTERLEAVE_BLOCK_SIZE == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaMalloc(&T, N * N * sizeof(struct gpu___mpz_t)));

	const gpu_mp_size_t Trowstride = N;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);
	SPLIT_MATRIX_NUM(p, N, N, T, Trowstride, 1, 2, 3, 4);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_strided_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define SUB(c_, cr_, a_, ar_, b_, br_) \
		sub_matrices_strided_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		winograd_strided_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define AML(c_, cr_, a_, ar_, b_, br_) \
		winograd_addmul_strided_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	WINOGRAD_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL
#undef AML

	CUSUC(cudaFree(T));
}

void winograd_addmul_strided_gpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	fprintf(stderr, "winograd_addmul_strided_gpu(%d)\n", N);

	assert(INTERLEAVE_BLOCK_SIZE <= Crowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Arowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Browstride);

	if (N <= BASE_SIZE) {
		launcher_addmul_strided
			<<<dim3((N + GPU_BLK_W - 1) / GPU_BLK_W, (N + GPU_BLK_H - 1) / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>
			(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		CUSUC(cudaDeviceSynchronize());
		return;
	}

	assert(N % 2 == 0);
	assert(N / 2 % INTERLEAVE_BLOCK_SIZE == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaMalloc(&T, N * N * sizeof(struct gpu___mpz_t)));

	const gpu_mp_size_t Trowstride = N;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);
	SPLIT_MATRIX_NUM(p, N, N, T, Trowstride, 1, 2, 3, 4);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_strided_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define SUB(c_, cr_, a_, ar_, b_, br_) \
		sub_matrices_strided_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		winograd_strided_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define AML(c_, cr_, a_, ar_, b_, br_) \
		winograd_addmul_strided_gpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	WINOGRAD_ADDMUL_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL
#undef AML

	CUSUC(cudaFree(T));
}

// vim: set sw=4 ts=4 noet:
