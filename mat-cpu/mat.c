#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include "mat.h"


static gmp_randstate_t g_rand_state;

__attribute__((constructor))
static void init_rand_state(void) {
	gmp_randinit_default(g_rand_state);
	struct timeval tv;
	gettimeofday(&tv, NULL);
	gmp_randseed_ui(g_rand_state, tv.tv_sec * 1000000U + tv.tv_usec);
}

__attribute__((destructor))
static void deinit_rand_state(void) {
	gmp_randclear(g_rand_state);
}


mat_t mat_init(unsigned int W, unsigned int H) {
	mat_t A = (mat_t)malloc(W * H * sizeof(mpz_t));
	for (unsigned int i = 0; i < W * H; i++) {
		mpz_init(A[i]);
	}
	return A;
}

void mat_free(unsigned int W, unsigned int H, mat_t A) {
	for (unsigned int i = 0; i < W * H; i++) {
		mpz_clear(A[i]);
	}
	free(A);
}

void mat_random(unsigned int W, unsigned int H, mat_t A, int nbits) {
	for (unsigned int i = 0; i < W * H; i++) {
		mpz_urandomb(A[i], g_rand_state, nbits);
		if (gmp_urandomb_ui(g_rand_state, 1) == 0) mpz_neg(A[i], A[i]);
	}
}

void mat_random_mpz(mpz_t z, int nbits) {
	mpz_urandomb(z, g_rand_state, nbits);
	if (gmp_urandomb_ui(g_rand_state, 1) == 0) mpz_neg(z, z);
}

void mat_seed_generator(unsigned long int seed) {
	gmp_randseed_ui(g_rand_state, seed);
}

void mat_add(unsigned int W, unsigned int H, mat_t C, mp_size_t Crowstride, mat_src_t A, mp_size_t Arowstride, mat_src_t B, mp_size_t Browstride) {
	// fprintf(stderr, "mat_add(%u, %u, %p, %p, %p)\n", W, H, C, A, B);
	for (unsigned int y = 0; y < H; y++) {
		for (unsigned int x = 0; x < W; x++) {
			mpz_add(C[Crowstride*y+x], A[Arowstride*y+x], B[Browstride*y+x]);
		}
	}
}

void mat_sub(unsigned int W, unsigned int H, mat_t C, mp_size_t Crowstride, mat_src_t A, mp_size_t Arowstride, mat_src_t B, mp_size_t Browstride) {
	// fprintf(stderr, "mat_sub(%u, %u, %p, %p, %p)\n", W, H, C, A, B);
	for (unsigned int y = 0; y < H; y++) {
		for (unsigned int x = 0; x < W; x++) {
			mpz_sub(C[Crowstride*y+x], A[Arowstride*y+x], B[Browstride*y+x]);
		}
	}
}

void mat_mul(unsigned int K, unsigned int M, unsigned int N, mat_t C, mp_size_t Crowstride, mat_src_t A, mp_size_t Arowstride, mat_src_t B, mp_size_t Browstride) {
	// fprintf(stderr, "mat_mul(%u, %u, %u, %p, %p, %p)\n", K, M, N, C, A, B);
	for (unsigned int y = 0; y < K; y++) {
		for (unsigned int x = 0; x < N; x++) {
			mpz_mul(C[Crowstride*y+x], A[Arowstride*y+0], B[Browstride*0+x]);
			for (unsigned int k = 1; k < M; k++) {
				mpz_addmul(C[Crowstride*y+x], A[Arowstride*y+k], B[Browstride*k+x]);
			}
		}
	}
}

void mat_addmul(unsigned int K, unsigned int M, unsigned int N, mat_t C, mp_size_t Crowstride, mat_src_t A, mp_size_t Arowstride, mat_src_t B, mp_size_t Browstride) {
	// fprintf(stderr, "mat_addmul(%u, %u, %u, %p, %p, %p)\n", K, M, N, C, A, B);
	for (unsigned int y = 0; y < K; y++) {
		for (unsigned int x = 0; x < N; x++) {
			for (unsigned int k = 0; k < M; k++) {
				mpz_addmul(C[Crowstride*y+x], A[Arowstride*y+k], B[Browstride*k+x]);
			}
		}
	}
}

static inline const mpz_t* submatrix_C(mp_size_t W, mp_size_t H, mat_src_t M, mp_size_t rowstride, mp_size_t ix, mp_size_t iy) {
	return M + iy * (H / 2 * rowstride) + ix * (W / 2);
}

static inline mat_t submatrix(mp_size_t W, mp_size_t H, mat_t M, mp_size_t rowstride, mp_size_t ix, mp_size_t iy) {
	return M + iy * (H / 2 * rowstride) + ix * (W / 2);
}

void mat_mul_strassen(
		unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int nlevels) {

	if (nlevels == 0) {
		mat_mul(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);

	// fprintf(stderr, "mat_mul_strassen(%u, %p, %p, %p, %d)\n", N, C, A, B, nlevels);

	mat_t T = mat_init(N, N);
	mp_size_t Trowstride = N;

	mat_src_t a11 = submatrix_C(N, N, A, Arowstride, 0, 0);
	mat_src_t a12 = submatrix_C(N, N, A, Arowstride, 1, 0);
	mat_src_t a21 = submatrix_C(N, N, A, Arowstride, 0, 1);
	mat_src_t a22 = submatrix_C(N, N, A, Arowstride, 1, 1);

	mat_src_t b11 = submatrix_C(N, N, B, Browstride, 0, 0);
	mat_src_t b12 = submatrix_C(N, N, B, Browstride, 1, 0);
	mat_src_t b21 = submatrix_C(N, N, B, Browstride, 0, 1);
	mat_src_t b22 = submatrix_C(N, N, B, Browstride, 1, 1);

	mat_t c11 = submatrix(N, N, C, Crowstride, 0, 0);
	mat_t c12 = submatrix(N, N, C, Crowstride, 1, 0);
	mat_t c21 = submatrix(N, N, C, Crowstride, 0, 1);
	mat_t c22 = submatrix(N, N, C, Crowstride, 1, 1);

	mat_t p1 = submatrix(N, N, T, Trowstride, 0, 0);
	mat_t p2 = submatrix(N, N, T, Trowstride, 1, 0);
	mat_t p3 = submatrix(N, N, T, Trowstride, 0, 1);

#define ADD(c_, cr_, a_, ar_, b_, br_) mat_add(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)
#define SUB(c_, cr_, a_, ar_, b_, br_) mat_sub(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)
#define MUL(c_, cr_, a_, ar_, b_, br_) mat_mul_strassen(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, nlevels - 1)

	// 3 temporaries version
	ADD(p1,  T, a11, A, a22, A);
	ADD(p2,  T, b11, B, b22, B);
	MUL(c11, C, p1,  T, p2,  T);

	SUB(p1,  T, b12, B, b22, B);
	MUL(c12, C, a11, A, p1,  T);

	ADD(c22, C, c11, C, c12, C);

	ADD(p1,  T, a21, A, a22, A);
	MUL(c21, C, p1,  T, b11, B);

	SUB(c22, C, c22, C, c21, C);

	SUB(p2,  T, a12, A, a22, A);
	ADD(p3,  T, b21, B, b22, B);
	MUL(p1,  T, p2,  T, p3,  T);

	ADD(c11, C, c11, C, p1,  T);

	SUB(p2,  T, a21, A, a11, A);
	ADD(p3,  T, b11, B, b12, B);
	MUL(p1,  T, p2,  T, p3,  T);

	ADD(c22, C, c22, C, p1,  T);

	SUB(p2,  T, b21, B, b11, B);
	MUL(p1,  T, a22, A, p2,  T);

	ADD(c11, C, c11, C, p1,  T);

	ADD(c21, C, c21, C, p1,  T);

	ADD(p2,  T, a11, A, a12, A);
	MUL(p1,  T, p2,  T, b22, B);

	SUB(c11, C, c11, C, p1,  T);

	ADD(c12, C, c12, C, p1,  T);

#undef ADD
#undef SUB
#undef MUL

	mat_free(N, N, T);
}

void mat_mul_winograd(
		unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int nlevels) {

	if (nlevels == 0) {
		mat_mul(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);

	mat_t T = mat_init(N, N);
	mp_size_t Trowstride = N;

	mat_src_t a11 = submatrix_C(N, N, A, Arowstride, 0, 0);
	mat_src_t a12 = submatrix_C(N, N, A, Arowstride, 1, 0);
	mat_src_t a21 = submatrix_C(N, N, A, Arowstride, 0, 1);
	mat_src_t a22 = submatrix_C(N, N, A, Arowstride, 1, 1);

	mat_src_t b11 = submatrix_C(N, N, B, Browstride, 0, 0);
	mat_src_t b12 = submatrix_C(N, N, B, Browstride, 1, 0);
	mat_src_t b21 = submatrix_C(N, N, B, Browstride, 0, 1);
	mat_src_t b22 = submatrix_C(N, N, B, Browstride, 1, 1);

	mat_t c11 = submatrix(N, N, C, Crowstride, 0, 0);
	mat_t c12 = submatrix(N, N, C, Crowstride, 1, 0);
	mat_t c21 = submatrix(N, N, C, Crowstride, 0, 1);
	mat_t c22 = submatrix(N, N, C, Crowstride, 1, 1);

	mat_t p1 = submatrix(N, N, T, Trowstride, 0, 0);
	mat_t p2 = submatrix(N, N, T, Trowstride, 1, 0);
	mat_t p3 = submatrix(N, N, T, Trowstride, 0, 1);
	// mat_t p4 = submatrix(N, N, T, Trowstride, 1, 1);

#define ADD(c_, cr_, a_, ar_, b_, br_) mat_add(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)
#define SUB(c_, cr_, a_, ar_, b_, br_) mat_sub(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)
#define MUL(c_, cr_, a_, ar_, b_, br_) mat_mul_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, nlevels - 1)
#define ADDMUL(c_, cr_, a_, ar_, b_, br_) mat_addmul_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, nlevels - 1)

	ADD(   p1,  T, a21, A, a22, A);
	SUB(   p2,  T, b12, B, b11, B);
	MUL(   c22, C, p1,  T, p2,  T);
	SUB(   p1,  T, p1,  T, a11, A);
	SUB(   p2,  T, b22, B, p2,  T);
	MUL(   p3,  T, a11, A, b11, B);
	MUL(   c11, C, a12, A, b21, B);
	ADD(   c11, C, c11, C, p3,  T);
	ADDMUL(p3,  T, p1,  T, p2,  T);
	SUB(   p1,  T, a12, A, p1,  T);
	SUB(   p2,  T, b21, B, p2,  T);
	MUL(   c12, C, p1,  T, b22, B);
	ADD(   c12, C, c12, C, c22, C);
	ADD(   c12, C, c12, C, p3,  T);
	MUL(   c21, C, a22, A, p2,  T);
	SUB(   p1,  T, a11, A, a21, A);
	SUB(   p2,  T, b22, B, b12, B);
	ADDMUL(p3,  T, p1,  T, p2,  T);
	ADD(   c21, C, c21, C, p3,  T);
	ADD(   c22, C, c22, C, p3,  T);

#undef ADD
#undef SUB
#undef MUL
#undef ADDMUL

	mat_free(N, N, T);
}

void mat_addmul_winograd(
		unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int nlevels) {

	if (nlevels == 0) {
		mat_addmul(N, N, N, C, Crowstride, A, Arowstride, B, Browstride);
		return;
	}

	assert(N % 2 == 0);

	mat_t T = mat_init(N, N);
	mp_size_t Trowstride = N;

	mat_src_t a11 = submatrix_C(N, N, A, Arowstride, 0, 0);
	mat_src_t a12 = submatrix_C(N, N, A, Arowstride, 1, 0);
	mat_src_t a21 = submatrix_C(N, N, A, Arowstride, 0, 1);
	mat_src_t a22 = submatrix_C(N, N, A, Arowstride, 1, 1);

	mat_src_t b11 = submatrix_C(N, N, B, Browstride, 0, 0);
	mat_src_t b12 = submatrix_C(N, N, B, Browstride, 1, 0);
	mat_src_t b21 = submatrix_C(N, N, B, Browstride, 0, 1);
	mat_src_t b22 = submatrix_C(N, N, B, Browstride, 1, 1);

	mat_t c11 = submatrix(N, N, C, Crowstride, 0, 0);
	mat_t c12 = submatrix(N, N, C, Crowstride, 1, 0);
	mat_t c21 = submatrix(N, N, C, Crowstride, 0, 1);
	mat_t c22 = submatrix(N, N, C, Crowstride, 1, 1);

	mat_t p1 = submatrix(N, N, T, Trowstride, 0, 0);
	mat_t p2 = submatrix(N, N, T, Trowstride, 1, 0);
	mat_t p3 = submatrix(N, N, T, Trowstride, 0, 1);
	// mat_t p4 = submatrix(N, N, T, Trowstride, 1, 1);

#define ADD(c_, cr_, a_, ar_, b_, br_) mat_add(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)
#define SUB(c_, cr_, a_, ar_, b_, br_) mat_sub(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride)
#define MUL(c_, cr_, a_, ar_, b_, br_) mat_mul_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, nlevels - 1)
#define ADDMUL(c_, cr_, a_, ar_, b_, br_) mat_addmul_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, nlevels - 1)

	ADD(   p1,  T, a21, A, a22, A);
	SUB(   p2,  T, b12, B, b11, B);
	MUL(   p3,  T, p1,  T, p2,  T);
	ADD(   c12, C, c12, C, p3,  T);
	ADD(   c22, C, c22, C, p3,  T);
	SUB(   p1,  T, p1,  T, a11, A);
	SUB(   p2,  T, b22, B, p2,  T);
	MUL(   p3,  T, a11, A, b11, B);
	ADD(   c11, C, c11, C, p3,  T);
	ADDMUL(p3,  T, p1,  T, p2,  T);
	ADDMUL(c11, C, a12, A, b21, B);
	SUB(   p1,  T, a12, A, p1,  T);
	SUB(   p2,  T, b21, B, p2,  T);
	ADDMUL(c12, C, p1,  T, b22, B);
	ADD(   c12, C, c12, C, p3,  T);
	ADDMUL(c21, C, a22, A, p2,  T);
	SUB(   p1,  T, a11, A, a21, A);
	SUB(   p2,  T, b22, B, b12, B);
	ADDMUL(p3,  T, p1,  T, p2,  T);
	ADD(   c21, C, c21, C, p3,  T);
	ADD(   c22, C, c22, C, p3,  T);

#undef ADD
#undef SUB
#undef MUL
#undef ADDMUL

	mat_free(N, N, T);
}

void mat_read(unsigned int W, unsigned int H, FILE *f, mat_t A) {
	for (unsigned int y = 0; y < H; y++) {
		for (unsigned int x = 0; x < W; x++) {
			mpz_inp_str(A[W*y+x], f, 16);
		}
	}
}

void mat_write(unsigned int W, unsigned int H, FILE *f, mat_src_t A, mp_size_t Arowstride) {
	for (unsigned int y = 0; y < H; y++) {
		for (unsigned int x = 0; x < W; x++) {
			if (x != 0) fprintf(f, " ");
			mpz_out_str(f, 16, A[Arowstride*y+x]);
		}
		fprintf(f, "\n");
	}
	fprintf(f, "\n");
}

void mat_write_oneline(unsigned int W, unsigned int H, FILE *f, mat_src_t A, mp_size_t Arowstride) {
	for (unsigned int y = 0; y < H; y++) {
		for (unsigned int x = 0; x < W; x++) {
			if (x != 0) fprintf(f, " ");
			mpz_out_str(f, 16, A[Arowstride*y+x]);
		}
		fprintf(f, " ");
	}
	fprintf(f, "\n");
}

// vim: set sw=4 ts=4 noet:
