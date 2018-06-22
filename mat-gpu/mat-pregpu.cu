#include <assert.h>
#include "mat-pregpu.h"
#include "mat-params.h"
#include "mat-defines.h"
#include "mat-gpu.h"
#include "cusuc.h"
#include "util.h"


template <void (*func)(gpu_mp_size_t, gpu_mp_size_t, gpu_mp_size_t, struct gpu___mpz_t*, gpu_mp_size_t, const struct gpu___mpz_t*, gpu_mp_size_t, const struct gpu___mpz_t*, gpu_mp_size_t),
          bool transfer_C>
static void transfer_recurse(
		gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	struct gpu___mpz_t *devA, *devB, *devC;
	CUSUC(cudaMalloc(&devA, M * K * sizeof(struct gpu___mpz_t)));
	CUSUC(cudaMalloc(&devB, N * M * sizeof(struct gpu___mpz_t)));
	CUSUC(cudaMalloc(&devC, N * K * sizeof(struct gpu___mpz_t)));

	CUSUC(cudaMemcpy2D(
				devA, M * sizeof(struct gpu___mpz_t),
				A, Arowstride * sizeof(struct gpu___mpz_t),
				M * sizeof(struct gpu___mpz_t), K,
				cudaMemcpyHostToDevice));
	CUSUC(cudaMemcpy2D(
				devB, N * sizeof(struct gpu___mpz_t),
				B, Browstride * sizeof(struct gpu___mpz_t),
				N * sizeof(struct gpu___mpz_t), M,
				cudaMemcpyHostToDevice));
	if (transfer_C) {
		CUSUC(cudaMemcpy2D(
					devC, N * sizeof(struct gpu___mpz_t),
					C, Crowstride * sizeof(struct gpu___mpz_t),
					N * sizeof(struct gpu___mpz_t), K,
					cudaMemcpyHostToDevice));
	}

	func(K, M, N, devC, N, devA, M, devB, N);

	CUSUC(cudaMemcpy2D(
				C, Crowstride * sizeof(struct gpu___mpz_t),
				devC, N * sizeof(struct gpu___mpz_t),
				N * sizeof(struct gpu___mpz_t), K,
				cudaMemcpyDeviceToHost));

	CUSUC(cudaFree(devA));
	CUSUC(cudaFree(devB));
	CUSUC(cudaFree(devC));
}

template <void (*func)(gpu_mp_size_t, struct gpu___mpz_t*, gpu_mp_size_t, const struct gpu___mpz_t*, gpu_mp_size_t, const struct gpu___mpz_t*, gpu_mp_size_t),
          bool transfer_C>
static void transfer_recurse(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	struct gpu___mpz_t *devA, *devB, *devC;
	CUSUC(cudaMalloc(&devA, N * N * sizeof(struct gpu___mpz_t)));
	CUSUC(cudaMalloc(&devB, N * N * sizeof(struct gpu___mpz_t)));
	CUSUC(cudaMalloc(&devC, N * N * sizeof(struct gpu___mpz_t)));

	CUSUC(cudaMemcpy2D(
				devA, N * sizeof(struct gpu___mpz_t),
				A, Arowstride * sizeof(struct gpu___mpz_t),
				N * sizeof(struct gpu___mpz_t), N,
				cudaMemcpyHostToDevice));
	CUSUC(cudaMemcpy2D(
				devB, N * sizeof(struct gpu___mpz_t),
				B, Browstride * sizeof(struct gpu___mpz_t),
				N * sizeof(struct gpu___mpz_t), N,
				cudaMemcpyHostToDevice));
	if (transfer_C) {
		CUSUC(cudaMemcpy2D(
					devC, N * sizeof(struct gpu___mpz_t),
					C, Crowstride * sizeof(struct gpu___mpz_t),
					N * sizeof(struct gpu___mpz_t), N,
					cudaMemcpyHostToDevice));
	}

	func(N, devC, N, devA, N, devB, N);

	CUSUC(cudaMemcpy2D(
				C, Crowstride * sizeof(struct gpu___mpz_t),
				devC, N * sizeof(struct gpu___mpz_t),
				N * sizeof(struct gpu___mpz_t), N,
				cudaMemcpyDeviceToHost));

	CUSUC(cudaFree(devA));
	CUSUC(cudaFree(devB));
	CUSUC(cudaFree(devC));
}


void naive_recursive_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	if (N <= TRANSFER_SIZE) {
		transfer_recurse<naive_recursive_gpu, false>(N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, (N / 2) * (N / 2) * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

	const gpu_mp_size_t Trowstride = N / 2;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);

	// Here, and in other places, add_matrices_cpu (and sub_matrices_gpu) are called with CPU pointers
	// as arguments. This seems incredibly suboptimal, and apparently does work; however, since the
	// additions and subtractions form such a small percentage of the actual runtime, I didn't think
	// it is performance-critical to fix this problem.
#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		naive_recursive_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	NAIVE_RECURSIVE_SEQUENCE;

#undef ADD
#undef MUL

	CUSUC(cudaFreeHost(T));
}

void naive_recursive_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	assert(INTERLEAVE_BLOCK_SIZE <= Crowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Arowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Browstride);

	if (N <= TRANSFER_SIZE) {
		transfer_recurse<naive_recursive_strided_gpu, false>(N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);
	assert(N / 2 % INTERLEAVE_BLOCK_SIZE == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, (N / 2) * (N / 2) * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

	const gpu_mp_size_t Trowstride = N / 2;

	SPLIT_MATRIX_CONST(a, N, N, A, Arowstride);
	SPLIT_MATRIX_CONST(b, N, N, B, Browstride);
	SPLIT_MATRIX(c, N, N, C, Crowstride);

#define ADD(c_, cr_, a_, ar_, b_, br_) \
		add_matrices_strided_gpu(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define MUL(c_, cr_, a_, ar_, b_, br_) \
		naive_recursive_strided_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	NAIVE_RECURSIVE_SEQUENCE;

#undef ADD
#undef MUL

	CUSUC(cudaFreeHost(T));
}

void strassen_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	if (N <= TRANSFER_SIZE) {
		transfer_recurse<strassen_gpu, false>(N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, N * N * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

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
		strassen_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	STRASSEN_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL

	CUSUC(cudaFreeHost(T));
}

void strassen_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	assert(INTERLEAVE_BLOCK_SIZE <= Crowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Arowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Browstride);

	if (N <= TRANSFER_SIZE) {
		transfer_recurse<strassen_strided_gpu, false>(N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);
	assert(N / 2 % INTERLEAVE_BLOCK_SIZE == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, N * N * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

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
		strassen_strided_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	STRASSEN_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL

	CUSUC(cudaFreeHost(T));
}

void winograd_addmul_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	if (N <= TRANSFER_SIZE) {
		transfer_recurse<winograd_gpu, false>(N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, N * N * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

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
		winograd_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define AML(c_, cr_, a_, ar_, b_, br_) \
		winograd_addmul_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	WINOGRAD_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL
#undef AML

	CUSUC(cudaFreeHost(T));
}

void winograd_addmul_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	if (N <= TRANSFER_SIZE) {
		transfer_recurse<winograd_addmul_gpu, true>(N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, N * N * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

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
		winograd_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define AML(c_, cr_, a_, ar_, b_, br_) \
		winograd_addmul_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	WINOGRAD_ADDMUL_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL
#undef AML

	CUSUC(cudaFreeHost(T));
}

void winograd_addmul_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride);

void winograd_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	assert(INTERLEAVE_BLOCK_SIZE <= Crowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Arowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Browstride);

	if (N <= TRANSFER_SIZE) {
		transfer_recurse<winograd_strided_gpu, false>(N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);
	assert(N / 2 % INTERLEAVE_BLOCK_SIZE == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, N * N * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

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
		winograd_strided_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define AML(c_, cr_, a_, ar_, b_, br_) \
		winograd_addmul_strided_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	WINOGRAD_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL
#undef AML

	CUSUC(cudaFreeHost(T));
}

void winograd_addmul_strided_pregpu(
		gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	assert(INTERLEAVE_BLOCK_SIZE <= Crowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Arowstride);
	assert(INTERLEAVE_BLOCK_SIZE <= Browstride);

	if (N <= TRANSFER_SIZE) {
		transfer_recurse<winograd_addmul_strided_gpu, true>(N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);
	assert(N / 2 % INTERLEAVE_BLOCK_SIZE == 0);

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, N * N * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

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
		winograd_strided_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

#define AML(c_, cr_, a_, ar_, b_, br_) \
		winograd_addmul_strided_pregpu(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)

	WINOGRAD_ADDMUL_SEQUENCE;

#undef ADD
#undef SUB
#undef MUL
#undef AML

	CUSUC(cudaFreeHost(T));
}

// Variadic templates are easily the most C++'y feature in this codebase
template <typename Ftype, typename ...FAtypes>
static void naive_smallrect_recurse(
		gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride,
		gpu_mp_size_t ltsz,
		Ftype mul_callback, FAtypes ...callback_args) {

	struct gpu___mpz_t *T;
	CUSUC(cudaHostAlloc(&T, N * K * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

	const gpu_mp_size_t Trowstride = N;

	const struct gpu___mpz_t *a11 = submatrix_ltwh(A, Arowstride, ltsz, ltsz, 0, 0);
	const struct gpu___mpz_t *a12 = submatrix_ltwh(A, Arowstride, ltsz, ltsz, 1, 0);
	const struct gpu___mpz_t *a21 = submatrix_ltwh(A, Arowstride, ltsz, ltsz, 0, 1);
	const struct gpu___mpz_t *a22 = submatrix_ltwh(A, Arowstride, ltsz, ltsz, 1, 1);

	const struct gpu___mpz_t *b11 = submatrix_ltwh(B, Browstride, ltsz, ltsz, 0, 0);
	const struct gpu___mpz_t *b12 = submatrix_ltwh(B, Browstride, ltsz, ltsz, 1, 0);
	const struct gpu___mpz_t *b21 = submatrix_ltwh(B, Browstride, ltsz, ltsz, 0, 1);
	const struct gpu___mpz_t *b22 = submatrix_ltwh(B, Browstride, ltsz, ltsz, 1, 1);

	struct gpu___mpz_t *c11 = submatrix_ltwh(C, Crowstride, ltsz, ltsz, 0, 0);
	struct gpu___mpz_t *c12 = submatrix_ltwh(C, Crowstride, ltsz, ltsz, 1, 0);
	struct gpu___mpz_t *c21 = submatrix_ltwh(C, Crowstride, ltsz, ltsz, 0, 1);
	struct gpu___mpz_t *c22 = submatrix_ltwh(C, Crowstride, ltsz, ltsz, 1, 1);

	struct gpu___mpz_t *p1 = submatrix_ltwh(T, Trowstride, ltsz, ltsz, 0, 0);
	struct gpu___mpz_t *p2 = submatrix_ltwh(T, Trowstride, ltsz, ltsz, 1, 0);
	struct gpu___mpz_t *p3 = submatrix_ltwh(T, Trowstride, ltsz, ltsz, 0, 1);
	struct gpu___mpz_t *p4 = submatrix_ltwh(T, Trowstride, ltsz, ltsz, 1, 1);

	// NOTE: ADD takes (h,w), not (w,h)!
#define ADD(h_, w_,     c_, cr_, a_, ar_, b_, br_) add_matrices_gpu(w_, h_, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)
#define MUL(k_, m_, n_, c_, cr_, a_, ar_, b_, br_) mul_callback(k_, m_, n_, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, callback_args...)

	const bool tK = ltsz < K, tM = ltsz < M, tN = ltsz < N;

	if (true          ) MUL(  ltsz,   ltsz,   ltsz, c11, C, a11, A, b11, B);
	if (      tM      ) MUL(  ltsz, M-ltsz,   ltsz, p1,  T, a12, A, b21, B);
	if (true          ) ADD(  ltsz,           ltsz, c11, C, c11, C, p1,  T);

	if (tN            ) MUL(  ltsz,   ltsz, N-ltsz, c12, C, a11, A, b12, B);
	if (tN && tM      ) MUL(  ltsz, M-ltsz, N-ltsz, p2,  T, a12, A, b22, B);
	if (tN            ) ADD(  ltsz,         N-ltsz, c12, C, c12, C, p2,  T);

	if (tK            ) MUL(K-ltsz,   ltsz,   ltsz, c21, C, a21, A, b11, B);
	if (tK && tM      ) MUL(K-ltsz, M-ltsz,   ltsz, p3,  T, a22, A, b21, B);
	if (tK            ) ADD(K-ltsz,           ltsz, c21, C, c21, C, p3,  T);

	if (tK &&       tN) MUL(K-ltsz,   ltsz, N-ltsz, c22, C, a21, A, b12, B);
	if (tK && tM && tN) MUL(K-ltsz, M-ltsz, N-ltsz, p4,  T, a22, A, b22, B);
	if (tK && tN      ) ADD(K-ltsz,         N-ltsz, c22, C, c22, C, p4,  T);

#undef ADD
#undef MUL

	CUSUC(cudaFreeHost(T));
}

void naive_smallrect_pregpu(
		gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	if (K == 0 || M == 0 || N == 0) return;

	if (K <= BASE_SIZE && M <= BASE_SIZE && N <= BASE_SIZE) {
		transfer_recurse<naive_smallrect_gpu, false>(K, M, N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	naive_smallrect_recurse(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, min(min(K, M), N), naive_smallrect_pregpu);
}

void squarify_mul_pregpu(
		gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
		struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
		const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
		const struct gpu___mpz_t *B, gpu_mp_size_t Browstride,
		void (*squarefunc)(
			gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crs,
			const struct gpu___mpz_t *A, gpu_mp_size_t Ars,
			const struct gpu___mpz_t *B, gpu_mp_size_t Brs)) {

	if (K == 0 || M == 0 || N == 0) {
		return;
	}

	if (K <= BASE_SIZE || M <= BASE_SIZE || M <= BASE_SIZE || N <= BASE_SIZE) {
		naive_smallrect_pregpu(K, M, N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}
	if (K == M && M == N && remove_factors(K, 2) <= BASE_SIZE) {
		return squarefunc(K, C, Crowstride, A, Arowstride, B, Browstride);
	}

	gpu_mp_size_t mindim = min(min(K, M), N);
	assert(mindim > 0);
	gpu_mp_size_t ptwo = 1 << (8 * sizeof(gpu_mp_size_t) - __builtin_clz(mindim) - 1);

	naive_smallrect_recurse(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, ptwo, squarify_mul_pregpu, squarefunc);
}

// vim: set sw=4 ts=4 noet:
