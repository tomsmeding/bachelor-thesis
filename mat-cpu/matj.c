#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "matj.h"
#include "mat.h"
#include "util.h"


// #define DEBUG


static inline int min(int a, int b) { return a < b ? a : b; }


struct jdata {
	union {
		struct { unsigned int K, M, N; };
		struct { unsigned int W, H; };
	};
	mat_t C;
	mat_src_t A, B;
	mp_size_t Crowstride, Arowstride, Browstride;
};

struct jdata* jdata_makeWH(
		unsigned int W, unsigned int H,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride) {

	struct jdata *jdata = malloc(sizeof(struct jdata));
	jdata->W = W; jdata->H = H;
	jdata->C = C; jdata->A = A; jdata->B = B;
	jdata->Crowstride = Crowstride; jdata->Arowstride = Arowstride; jdata->Browstride = Browstride;
	return jdata;
}

struct jdata* jdata_makeKMN(
		unsigned int K, unsigned int M, unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride) {

	struct jdata *jdata = malloc(sizeof(struct jdata));
	jdata->K = K; jdata->M = M; jdata->N = N;
	jdata->C = C; jdata->A = A; jdata->B = B;
	jdata->Crowstride = Crowstride; jdata->Arowstride = Arowstride; jdata->Browstride = Browstride;
	return jdata;
}

#define JDATA_PACK_WH() (jdata_makeWH(W, H, C, Crowstride, A, Arowstride, B, Browstride))
#define JDATA_PACK_KMN() (jdata_makeKMN(K, M, N, C, Crowstride, A, Arowstride, B, Browstride))

#define JDATA_UNPACK_WH(j_) \
	unsigned int W = (j_)->W, H = (j_)->H; mat_src_t A = (j_)->A, B = (j_)->B; mat_t C = (j_)->C; \
	mp_size_t Crowstride = (j_)->Crowstride, Arowstride = (j_)->Arowstride, Browstride = (j_)->Browstride; \
	free(j_);

#define JDATA_UNPACK_KMN(j_) \
	unsigned int K = (j_)->K, M = (j_)->M, N = (j_)->N; mat_src_t A = (j_)->A, B = (j_)->B; mat_t C = (j_)->C; \
	mp_size_t Crowstride = (j_)->Crowstride, Arowstride = (j_)->Arowstride, Browstride = (j_)->Browstride; \
	free(j_);


// Used for recursive algorithms
struct jdata_ext {
	unsigned int K, M, N;
	mat_t C;
	mat_src_t A, B;
	mp_size_t Crowstride, Arowstride, Browstride;
	unsigned int nlevels;
};

struct jdata_ext* jdata_ext_make(
		unsigned int K, unsigned int M, unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int nlevels) {

	struct jdata_ext *jdata = malloc(sizeof(struct jdata_ext));
	jdata->K = K; jdata->M = M; jdata->N = N;
	jdata->C = C; jdata->A = A; jdata->B = B;
	jdata->Crowstride = Crowstride; jdata->Arowstride = Arowstride; jdata->Browstride = Browstride;
	jdata->nlevels = nlevels;
	return jdata;
}

#define JDATA_EXT_PACK_KMN() (jdata_ext_make(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, nlevels))
#define JDATA_EXT_UNPACK_KMN(j_) \
	unsigned int K = (j_)->K, M = (j_)->M, N = (j_)->N; mat_src_t A = (j_)->A, B = (j_)->B; mat_t C = (j_)->C; \
	mp_size_t Crowstride = (j_)->Crowstride, Arowstride = (j_)->Arowstride, Browstride = (j_)->Browstride; \
	unsigned int nlevels = (j_)->nlevels; \
	free(j_);


static void matj_add_j(void *jdata) {
	JDATA_UNPACK_WH((struct jdata*)jdata);
	mat_add(W, H, C, Crowstride, A, Arowstride, B, Browstride);
#ifdef DEBUG
	fprintf(stderr, "@ ADD %ux%u ", W, H); mat_write_oneline(W, H, stderr, C, Crowstride);
#endif
}

static void matj_sub_j(void *jdata) {
	JDATA_UNPACK_WH((struct jdata*)jdata);
	mat_sub(W, H, C, Crowstride, A, Arowstride, B, Browstride);
#ifdef DEBUG
	fprintf(stderr, "@ SUB %ux%u ", W, H); mat_write_oneline(W, H, stderr, C, Crowstride);
#endif
}

static void matj_mul_j(void *jdata) {
	JDATA_UNPACK_KMN((struct jdata*)jdata);
	mat_mul(K, M, N, C, Crowstride, A, Arowstride, B, Browstride);
#ifdef DEBUG
	fprintf(stderr, "@ MUL %ux%ux%u ", K, M, N); mat_write_oneline(N, K, stderr, C, Crowstride);
#endif
}

static void matj_addmul_j(void *jdata) {
	JDATA_UNPACK_KMN((struct jdata*)jdata);
	mat_addmul(K, M, N, C, Crowstride, A, Arowstride, B, Browstride);
#ifdef DEBUG
	fprintf(stderr, "@ ADDMUL %ux%ux%u ", K, M, N); mat_write_oneline(N, K, stderr, C, Crowstride);
#endif
}

job_t matj_add(unsigned int W, unsigned int H, mat_t C, mp_size_t Crowstride, mat_src_t A, mp_size_t Arowstride, mat_src_t B, mp_size_t Browstride, struct job_dep_list_t gdeps) {
	if (W == 0 || H == 0) return -1;
#ifdef DEBUG
	fprintf(stderr, "matj_add(%u, %u, %p, %p, %p)\n", W, H, C, A, B);
#endif
	return job_submit(gdeps, matj_add_j, JDATA_PACK_WH());
}

job_t matj_sub(unsigned int W, unsigned int H, mat_t C, mp_size_t Crowstride, mat_src_t A, mp_size_t Arowstride, mat_src_t B, mp_size_t Browstride, struct job_dep_list_t gdeps) {
	if (W == 0 || H == 0) return -1;
#ifdef DEBUG
	fprintf(stderr, "matj_sub(%u, %u, %p, %p, %p)\n", W, H, C, A, B);
#endif
	return job_submit(gdeps, matj_sub_j, JDATA_PACK_WH());
}

job_t matj_mul(unsigned int K, unsigned int M, unsigned int N, mat_t C, mp_size_t Crowstride, mat_src_t A, mp_size_t Arowstride, mat_src_t B, mp_size_t Browstride, struct job_dep_list_t gdeps) {
	if (K == 0 || M == 0 || N == 0) return -1;
#ifdef DEBUG
	fprintf(stderr, "matj_mul(%u, %u, %u, %p, %p, %p)\n", K, M, N, C, A, B);
#endif
	return job_submit(gdeps, matj_mul_j, JDATA_PACK_KMN());
}

job_t matj_addmul(unsigned int K, unsigned int M, unsigned int N, mat_t C, mp_size_t Crowstride, mat_src_t A, mp_size_t Arowstride, mat_src_t B, mp_size_t Browstride, struct job_dep_list_t gdeps) {
	if (K == 0 || M == 0 || N == 0) return -1;
#ifdef DEBUG
	fprintf(stderr, "matj_addmul(%u, %u, %u, %p, %p, %p)\n", K, M, N, C, A, B);
#endif
	return job_submit(gdeps, matj_addmul_j, JDATA_PACK_KMN());
}

static void matj_mul_strassen_j(void *jdata_e_) {
	JDATA_EXT_UNPACK_KMN((struct jdata_ext*)jdata_e_);
	assert(K == N && M == N);
	mat_mul_strassen(N, C, Crowstride, A, Arowstride, B, Browstride, nlevels);
}

static void matj_mul_winograd_j(void *jdata_e_) {
	JDATA_EXT_UNPACK_KMN((struct jdata_ext*)jdata_e_);
	assert(K == N && M == N);
	mat_mul_winograd(N, C, Crowstride, A, Arowstride, B, Browstride, nlevels);
}

static void matj_addmul_winograd_j(void *jdata_e_) {
	JDATA_EXT_UNPACK_KMN((struct jdata_ext*)jdata_e_);
	assert(K == N && M == N);
	mat_addmul_winograd(N, C, Crowstride, A, Arowstride, B, Browstride, nlevels);
}

static inline const mpz_t* submatrix_C_ltwh(mat_src_t M, mp_size_t rowstride, mp_size_t ltw, mp_size_t lth, mp_size_t ix, mp_size_t iy) {
	return M + iy * (lth * rowstride) + ix * ltw;
}

static inline mat_t submatrix_ltwh(mat_t M, mp_size_t rowstride, mp_size_t ltw, mp_size_t lth, mp_size_t ix, mp_size_t iy) {
	return M + iy * (lth * rowstride) + ix * ltw;
}

static inline const mpz_t* submatrix_C(mp_size_t W, mp_size_t H, mat_src_t M, mp_size_t rowstride, mp_size_t ix, mp_size_t iy) {
	return submatrix_C_ltwh(M, rowstride, W / 2, H / 2, ix, iy);
}

static inline mat_t submatrix(mp_size_t W, mp_size_t H, mat_t M, mp_size_t rowstride, mp_size_t ix, mp_size_t iy) {
	return submatrix_ltwh(M, rowstride, W / 2, H / 2, ix, iy);
}

struct jdata_strassen {
	unsigned int N;
	mat_t T, T2;
#ifdef DEBUG
	mat_t C;
	mp_size_t Crowstride;
#endif
};

static void strassen_cleanup_j(void *jdata_) {
	struct jdata_strassen *jdata = (struct jdata_strassen*)jdata_;
#ifdef DEBUG
	fprintf(stderr, "@ STR %u ", jdata->N); mat_write_oneline(jdata->N, jdata->N, stderr, jdata->C, jdata->Crowstride);
#endif
	mat_free(jdata->N, jdata->N, jdata->T);
	mat_free(jdata->N, jdata->N, jdata->T2);
	free(jdata);
}

job_t matj_mul_strassen(
		unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps) {

	if (N == 0) return -1;

	if (N <= minjobsize) {
		const unsigned int K = N, M = N;  // for jdata packing
		const unsigned int nlevels = N <= minblocksize ? 0 : round_down_power2(N / minblocksize);
		if (nlevels == 0) return matj_mul(N, N, N, C, Crowstride, A, Arowstride, B, Browstride, gdeps);
		else return job_submit(gdeps, matj_mul_strassen_j, JDATA_EXT_PACK_KMN());
	}

	assert(N % 2 == 0);

#ifdef DEBUG
	fprintf(stderr, "matj_mul_strassen(%u, %p, %p, %p)\n", N, C, A, B);
#endif

	mat_t T = mat_init(N, N), T2 = mat_init(N, N);
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
	mat_t p4 = submatrix(N, N, T, Trowstride, 1, 1);
	mat_t p5 = submatrix(N, N, T2, Trowstride, 0, 0);
	mat_t p6 = submatrix(N, N, T2, Trowstride, 1, 0);
	mat_t p7 = submatrix(N, N, T2, Trowstride, 0, 1);
	mat_t p8 = submatrix(N, N, T2, Trowstride, 1, 1);

#define ADD(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_add(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, d_))
#define SUB(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_sub(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, d_))
#define MUL(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_mul_strassen(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, minblocksize, minjobsize, d_))

	struct job_dep_list_t j1, j2, j3, j4, j5, j6, j7, j8, j9, ja, jb, jc, jd, je, jf, jg, jh, ji, jj, jk, jl, jm, jn, jo, jp;

	// 8 temporaries version
	j1 = ADD(p5,  T, a11, A, a22, A, job_dep_merge(1, &gdeps));
	j2 = ADD(p6,  T, b11, B, b22, B, job_dep_merge(1, &gdeps));
	j3 = MUL(p1,  T, p5,  T, p6,  T, job_dep_merge(2, &j1, &j2));

	j4 = ADD(p7,  T, a21, A, a22, A, job_dep_merge(1, &gdeps));
	j5 = MUL(p2,  T, p7,  T, b11, B, job_dep_merge(1, &j4));

	j6 = SUB(p8,  T, b12, B, b22, B, job_dep_merge(1, &gdeps));
	j7 = MUL(p3,  T, a11, A, p8,  T, job_dep_merge(2, &gdeps, &j6));

	j8 = SUB(p5,  T, b21, B, b11, B, job_dep_merge(2, &gdeps, &j3/*write conflict*/));
	j9 = MUL(p4,  T, a22, A, p5,  T, job_dep_merge(2, &gdeps, &j8));

	ja = SUB(c11, C, a21, A, a11, A, job_dep_merge(1, &gdeps));
	jb = ADD(c12, C, b11, B, b12, B, job_dep_merge(1, &gdeps));
	jc = MUL(p6,  T, c11, C, c12, C, job_dep_merge(3, &ja, &jb, &j3/*write conflict*/));

	jd = SUB(c21, C, a12, A, a22, A, job_dep_merge(1, &gdeps));
	je = ADD(c22, C, b21, B, b22, B, job_dep_merge(1, &gdeps));
	jf = MUL(p7,  T, c21, C, c22, C, job_dep_merge(3, &jd, &je, &j5/*write conflict*/));

	jg = ADD(p8,  T, a11, A, a12, A, job_dep_merge(2, &gdeps, &j7/*write conflict*/));
	jh = MUL(p5,  T, p8,  T, b22, B, job_dep_merge(3, &gdeps, &jg, &j9/*write conflict*/));

	ji = ADD(c11, C, p1,  T, p4,  T, job_dep_merge(3, &j3, &j9, &jc/*write conflict*/));
	jj = ADD(c11, C, c11, C, p7,  T, job_dep_merge(2, &ji, &jf));
	jk = SUB(c11, C, c11, C, p5,  T, job_dep_merge(2, &jj, &jh));

	jl = ADD(c21, C, p2,  T, p4,  T, job_dep_merge(3, &j5, &j9, &jf/*write conflict*/));

	jm = ADD(c12, C, p3,  T, p5,  T, job_dep_merge(3, &j7, &jh, &jc/*write conflict*/));

	jn = ADD(c22, C, p1,  T, p3,  T, job_dep_merge(3, &j3, &j7, &jf/*write conflict*/));
	jo = SUB(c22, C, c22, C, p2,  T, job_dep_merge(2, &jn, &j5));
	jp = ADD(c22, C, c22, C, p6,  T, job_dep_merge(2, &jo, &jc));

#undef ADD
#undef SUB
#undef MUL

	job_dep_list_free(gdeps);

	struct job_dep_list_t jobs = job_dep_list(0);
	job_dep_list_append_listp(&jobs, &jk);
	job_dep_list_append_listp(&jobs, &jl);
	job_dep_list_append_listp(&jobs, &jm);
	job_dep_list_append_listp(&jobs, &jp);

	job_dep_list_free(j1); job_dep_list_free(j2); job_dep_list_free(j3); job_dep_list_free(j4);
	job_dep_list_free(j5); job_dep_list_free(j6); job_dep_list_free(j7); job_dep_list_free(j8);
	job_dep_list_free(j9); job_dep_list_free(ja); job_dep_list_free(jb); job_dep_list_free(jc);
	job_dep_list_free(jd); job_dep_list_free(je); job_dep_list_free(jf); job_dep_list_free(jg);
	job_dep_list_free(jh); job_dep_list_free(ji); job_dep_list_free(jj); job_dep_list_free(jk);
	job_dep_list_free(jl); job_dep_list_free(jm); job_dep_list_free(jn); job_dep_list_free(jo);
	job_dep_list_free(jp);

	struct jdata_strassen *jdata = malloc(sizeof(struct jdata_strassen));
	jdata->N = N;
	jdata->T = T; jdata->T2 = T2;
#ifdef DEBUG
	jdata->C = C; jdata->Crowstride = Crowstride;
#endif

	return job_submit(jobs, strassen_cleanup_j, jdata);
}

static void winograd_cleanup_j(void *jdata_) {
	struct jdata_strassen *jdata = (struct jdata_strassen*)jdata_;
#ifdef DEBUG
	fprintf(stderr, "@ WMU %u ", jdata->N); mat_write_oneline(jdata->N, jdata->N, stderr, jdata->C, jdata->Crowstride);
#endif
	mat_free(jdata->N, jdata->N, jdata->T);
	free(jdata);
}

job_t matj_mul_winograd(
		unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps) {

	if (N == 0) return -1;

	if (N <= minjobsize) {
		const unsigned int K = N, M = N;  // for jdata packing
		const unsigned int nlevels = N <= minblocksize ? 0 : round_down_power2(N / minblocksize);
		if (nlevels == 0) return matj_mul(N, N, N, C, Crowstride, A, Arowstride, B, Browstride, gdeps);
		else return job_submit(gdeps, matj_mul_winograd_j, JDATA_EXT_PACK_KMN());
	}

	assert(N % 2 == 0);

#ifdef DEBUG
	fprintf(stderr, "matj_mul_winograd(%u, %p, %p, %p)\n", N, C, A, B);
#endif

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

#define ADD(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_add(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, d_))
#define SUB(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_sub(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, d_))
#define MUL(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_mul_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, minblocksize, minjobsize, d_))
#define ADDMUL(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_addmul_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, minblocksize, minjobsize, d_))

	struct job_dep_list_t j1, j2, j3, j4, j5, j6, j7, j8, j9, ja, jb, jc, jd, je, jf, jg, jh, ji, jj, jk;

	// TODO shorten dependency chains
	j1 = ADD(   p1,  T, a21, A, a22, A, job_dep_merge(1, &gdeps));
	j2 = SUB(   p2,  T, b12, B, b11, B, job_dep_merge(1, &gdeps));
	j3 = MUL(   c22, C, p1,  T, p2,  T, job_dep_merge(2, &j1, &j2));
	j4 = SUB(   p1,  T, p1,  T, a11, A, job_dep_merge(3, &gdeps, &j1, &j3/*write conflict*/));
	j5 = SUB(   p2,  T, b22, B, p2,  T, job_dep_merge(3, &gdeps, &j2, &j3/*write conflict*/));
	j6 = MUL(   p3,  T, a11, A, b11, B, job_dep_merge(1, &gdeps));
	j7 = MUL(   c11, C, a12, A, b21, B, job_dep_merge(1, &gdeps));
	j8 = ADD(   c11, C, c11, C, p3,  T, job_dep_merge(2, &j7, &j6));
	j9 = ADDMUL(p3,  T, p1,  T, p2,  T, job_dep_merge(4, &j4, &j5, &j6, &j8/*write conflict*/));
	ja = SUB(   p1,  T, a12, A, p1,  T, job_dep_merge(3, &gdeps, &j4, &j9/*write conflict*/));
	jb = SUB(   p2,  T, b21, B, p2,  T, job_dep_merge(3, &gdeps, &j5, &j9/*write conflict*/));
	jc = MUL(   c12, C, p1,  T, b22, B, job_dep_merge(2, &gdeps, &ja));
	jd = ADD(   c12, C, c12, C, c22, C, job_dep_merge(2, &jc, &j3));
	je = ADD(   c12, C, c12, C, p3,  T, job_dep_merge(2, &jd, &j9));
	jf = MUL(   c21, C, a22, A, p2,  T, job_dep_merge(2, &gdeps, &jb));
	jg = SUB(   p1,  T, a11, A, a21, A, job_dep_merge(2, &gdeps, &jc/*write conflict*/));
	jh = SUB(   p2,  T, b22, B, b12, B, job_dep_merge(2, &gdeps, &jf/*write conflict*/));
	ji = ADDMUL(p3,  T, p1,  T, p2,  T, job_dep_merge(4, &j9, &jg, &jh, &je/*write conflict*/));
	jj = ADD(   c21, C, c21, C, p3,  T, job_dep_merge(2, &jf, &ji));
	jk = ADD(   c22, C, c22, C, p3,  T, job_dep_merge(3, &j3, &ji, &jd/*write conflict*/));

#undef ADD
#undef SUB
#undef MUL
#undef ADDMUL

	job_dep_list_free(gdeps);

	struct job_dep_list_t jobs = job_dep_list(0);
	job_dep_list_append_listp(&jobs, &j8);
	job_dep_list_append_listp(&jobs, &je);
	job_dep_list_append_listp(&jobs, &jj);
	job_dep_list_append_listp(&jobs, &jk);

	job_dep_list_free(j1); job_dep_list_free(j2); job_dep_list_free(j3); job_dep_list_free(j4);
	job_dep_list_free(j5); job_dep_list_free(j6); job_dep_list_free(j7); job_dep_list_free(j8);
	job_dep_list_free(j9); job_dep_list_free(ja); job_dep_list_free(jb); job_dep_list_free(jc);
	job_dep_list_free(jd); job_dep_list_free(je); job_dep_list_free(jf); job_dep_list_free(jg);
	job_dep_list_free(jh); job_dep_list_free(ji); job_dep_list_free(jj); job_dep_list_free(jk);

	struct jdata_strassen *jdata = malloc(sizeof(struct jdata_strassen));
	jdata->N = N;
	jdata->T = T;
#ifdef DEBUG
	jdata->C = C; jdata->Crowstride = Crowstride;
#endif

	return job_submit(jobs, winograd_cleanup_j, jdata);
}

static void winograd_addmul_cleanup_j(void *jdata_) {
	struct jdata_strassen *jdata = (struct jdata_strassen*)jdata_;
#ifdef DEBUG
	fprintf(stderr, "@ WAM %u ", jdata->N); mat_write_oneline(jdata->N, jdata->N, stderr, jdata->C, jdata->Crowstride);
#endif
	mat_free(jdata->N, jdata->N, jdata->T);
	free(jdata);
}

job_t matj_addmul_winograd(
		unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps) {

	if (N == 0) return -1;

	if (N <= minjobsize) {
		const unsigned int K = N, M = N;  // for jdata packing
		const unsigned int nlevels = N <= minblocksize ? 0 : round_down_power2(N / minblocksize);
		if (nlevels == 0) return matj_addmul(N, N, N, C, Crowstride, A, Arowstride, B, Browstride, gdeps);
		else return job_submit(gdeps, matj_addmul_winograd_j, JDATA_EXT_PACK_KMN());
	}

	assert(N % 2 == 0);

#ifdef DEBUG
	fprintf(stderr, "matj_addmul_winograd(%u, %p, %p, %p)\n", N, C, A, B);
#endif

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

#define ADD(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_add(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, d_))
#define SUB(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_sub(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, d_))
#define MUL(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_mul_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, minblocksize, minjobsize, d_))
#define ADDMUL(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_addmul_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, minblocksize, minjobsize, d_))

	struct job_dep_list_t j1, j2, j3, j4, j5, j6, j7, j8, j9, ja, jb, jc, jd, je, jf, jg, jh, ji, jj, jk, jl;

	// TODO shorten dependency chains
	j1 = ADD(   p1,  T, a21, A, a22, A, job_dep_merge(1, &gdeps));
	j2 = SUB(   p2,  T, b12, B, b11, B, job_dep_merge(1, &gdeps));
	j3 = MUL(   p3,  T, p1,  T, p2,  T, job_dep_merge(2, &j1, &j2));
	j4 = ADD(   c12, C, c12, C, p3,  T, job_dep_merge(2, &gdeps, &j3));
	j5 = ADD(   c22, C, c22, C, p3,  T, job_dep_merge(2, &gdeps, &j3));
	j6 = SUB(   p1,  T, p1,  T, a11, A, job_dep_merge(3, &gdeps, &j1, &j3/*write conflict*/));
	j7 = SUB(   p2,  T, b22, B, p2,  T, job_dep_merge(3, &gdeps, &j2, &j3/*write conflict*/));
	j8 = MUL(   p3,  T, a11, A, b11, B, job_dep_merge(3, &gdeps, &j4/*write conflict*/, &j5/*write conflict*/));
	j9 = ADD(   c11, C, c11, C, p3,  T, job_dep_merge(2, &gdeps, &j8));
	ja = ADDMUL(p3,  T, p1,  T, p2,  T, job_dep_merge(4, &j6, &j7, &j8, &j9/*write conflict*/));
	jb = ADDMUL(c11, C, a12, A, b21, B, job_dep_merge(2, &gdeps, &j9));
	jc = SUB(   p1,  T, a12, A, p1,  T, job_dep_merge(3, &gdeps, &j6, &ja/*write conflict*/));
	jd = SUB(   p2,  T, b21, B, p2,  T, job_dep_merge(3, &gdeps, &j7, &ja/*write conflict*/));
	je = ADDMUL(c12, C, p1,  T, b22, B, job_dep_merge(3, &gdeps, &jc, &j4));
	jf = ADD(   c12, C, c12, C, p3,  T, job_dep_merge(2, &je, &ja));
	jg = ADDMUL(c21, C, a22, A, p2,  T, job_dep_merge(2, &gdeps, &jd));
	jh = SUB(   p1,  T, a11, A, a21, A, job_dep_merge(2, &gdeps, &je/*write conflict*/));
	ji = SUB(   p2,  T, b22, B, b12, B, job_dep_merge(2, &gdeps, &jg/*write conflict*/));
	jj = ADDMUL(p3,  T, p1,  T, p2,  T, job_dep_merge(3, &jh, &ji, &jf/*write conflict*/));
	jk = ADD(   c21, C, c21, C, p3,  T, job_dep_merge(2, &jg, &jj));
	jl = ADD(   c22, C, c22, C, p3,  T, job_dep_merge(2, &j5, &jj));

#undef ADD
#undef SUB
#undef MUL
#undef ADDMUL

	job_dep_list_free(gdeps);

	struct job_dep_list_t jobs = job_dep_list(0);
	job_dep_list_append_listp(&jobs, &jb);
	job_dep_list_append_listp(&jobs, &jf);
	job_dep_list_append_listp(&jobs, &jk);
	job_dep_list_append_listp(&jobs, &jl);

	job_dep_list_free(j1); job_dep_list_free(j2); job_dep_list_free(j3); job_dep_list_free(j4);
	job_dep_list_free(j5); job_dep_list_free(j6); job_dep_list_free(j7); job_dep_list_free(j8);
	job_dep_list_free(j9); job_dep_list_free(ja); job_dep_list_free(jb); job_dep_list_free(jc);
	job_dep_list_free(jd); job_dep_list_free(je); job_dep_list_free(jf); job_dep_list_free(jg);
	job_dep_list_free(jh); job_dep_list_free(ji); job_dep_list_free(jj); job_dep_list_free(jk);
	job_dep_list_free(jl);

	struct jdata_strassen *jdata = malloc(sizeof(struct jdata_strassen));
	jdata->N = N;
	jdata->T = T;
#ifdef DEBUG
	jdata->C = C; jdata->Crowstride = Crowstride;
#endif

	return job_submit(jobs, winograd_addmul_cleanup_j, jdata);
}

static void naive_winograd_cleanup_j(void *jdata_) {
	struct jdata_strassen *jdata = (struct jdata_strassen*)jdata_;
#ifdef DEBUG
	fprintf(stderr, "@ NWM %u ", jdata->N); mat_write_oneline(jdata->N, jdata->N, stderr, jdata->C, jdata->Crowstride);
#endif
	mat_free(jdata->N, jdata->N, jdata->T);
	free(jdata);
}

job_t matj_mul_naive_winograd(
		unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps) {

	if (N == 0) return -1;

	if (N <= minjobsize) {
		const unsigned int K = N, M = N;  // for jdata packing
		const unsigned int nlevels = N <= minblocksize ? 0 : round_down_power2(N / minblocksize);
		if (nlevels == 0) return matj_mul(N, N, N, C, Crowstride, A, Arowstride, B, Browstride, gdeps);
		else return job_submit(gdeps, matj_mul_winograd_j, JDATA_EXT_PACK_KMN());
	}

	assert(N % 2 == 0);

#ifdef DEBUG
	fprintf(stderr, "matj_mul_naive_winograd(%u, %p, %p, %p)\n", N, C, A, B);
#endif

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
	mat_t p4 = submatrix(N, N, T, Trowstride, 1, 1);

#define ADD(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_add(N / 2, N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, d_))
#define MUL(c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_mul_naive_winograd(N / 2, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, minblocksize, minjobsize, d_))

	struct job_dep_list_t j1, j2, j3, j4, j5, j6, j7, j8, j9, ja, jb, jc;

	j1 = MUL(c11, C, a11, A, b11, B, job_dep_merge(1, &gdeps));
	j2 = MUL(p1,  T, a12, A, b21, B, job_dep_merge(1, &gdeps));
	j3 = ADD(c11, C, c11, C, p1,  T, job_dep_merge(2, &j1, &j2));

	j4 = MUL(c12, C, a11, A, b12, B, job_dep_merge(1, &gdeps));
	j5 = MUL(p2,  T, a12, A, b22, B, job_dep_merge(1, &gdeps));
	j6 = ADD(c12, C, c12, C, p2,  T, job_dep_merge(2, &j4, &j5));

	j7 = MUL(c21, C, a21, A, b11, B, job_dep_merge(1, &gdeps));
	j8 = MUL(p3,  T, a22, A, b21, B, job_dep_merge(1, &gdeps));
	j9 = ADD(c21, C, c21, C, p3,  T, job_dep_merge(2, &j7, &j8));

	ja = MUL(c22, C, a21, A, b12, B, job_dep_merge(1, &gdeps));
	jb = MUL(p4,  T, a22, A, b22, B, job_dep_merge(1, &gdeps));
	jc = ADD(c22, C, c22, C, p4,  T, job_dep_merge(2, &ja, &jb));

#undef ADD
#undef MUL

	job_dep_list_free(gdeps);

	struct job_dep_list_t jobs = job_dep_list(0);
	job_dep_list_append_listp(&jobs, &j3);
	job_dep_list_append_listp(&jobs, &j6);
	job_dep_list_append_listp(&jobs, &j9);
	job_dep_list_append_listp(&jobs, &jc);

	job_dep_list_free(j1); job_dep_list_free(j2); job_dep_list_free(j3); job_dep_list_free(j4);
	job_dep_list_free(j5); job_dep_list_free(j6); job_dep_list_free(j7); job_dep_list_free(j8);
	job_dep_list_free(j9); job_dep_list_free(ja); job_dep_list_free(jb); job_dep_list_free(jc);

	struct jdata_strassen *jdata = malloc(sizeof(struct jdata_strassen));
	jdata->N = N;
	jdata->T = T;
#ifdef DEBUG
	jdata->C = C; jdata->Crowstride = Crowstride;
#endif

	return job_submit(jobs, naive_winograd_cleanup_j, jdata);
}

struct jdata_squarify {
	unsigned int K, M, N;
	mat_t T;
#ifdef DEBUG
	mat_t C;
	mp_size_t Crowstride;
#endif
};

static void squarify_cleanup_j(void *jdata_) {
	struct jdata_squarify *jdata = (struct jdata_squarify*)jdata_;
#ifdef DEBUG
	fprintf(stderr, "@ SQU %ux%ux%u ", jdata->K, jdata->M, jdata->N); mat_write_oneline(jdata->N, jdata->K, stderr, jdata->C, jdata->Crowstride);
#endif
	mat_free(jdata->N, jdata->K, jdata->T);
	free(jdata);
}

job_t matj_mul_squarify(
		unsigned int K, unsigned int M, unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int basew, unsigned int baseh,
		unsigned int minblocksize, unsigned int minjobsize,
		job_t (*squarefunc)(
			unsigned int N,
			mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs,
			unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps),
		struct job_dep_list_t gdeps) {

#ifdef DEBUG
	fprintf(stderr, "matj_mul_squarify(%u, %u, %u, %p, %p, %p)\n", K, M, N, C, A, B);
#endif

	if (K == 0 || M == 0 || N == 0) {
		return -1;
	}

	if (K <= baseh || M <= basew || M <= baseh || N <= basew) {
		return matj_mul(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, gdeps);
	}
	if (K == M && M == N && remove_factors(K, 2) <= minblocksize) {
		return squarefunc(K, C, Crowstride, A, Arowstride, B, Browstride, minblocksize, minjobsize, gdeps);
	}

	unsigned int mindim = min(min(K, M), N);
	assert(mindim > 0);
	unsigned int ptwo = 1 << (8 * sizeof(unsigned int) - __builtin_clz(mindim) - 1);

	mat_t T = mat_init(N, K);
	mp_size_t Trowstride = N;

	mat_src_t a11 = submatrix_C_ltwh(A, Arowstride, ptwo, ptwo, 0, 0);
	mat_src_t a12 = submatrix_C_ltwh(A, Arowstride, ptwo, ptwo, 1, 0);
	mat_src_t a21 = submatrix_C_ltwh(A, Arowstride, ptwo, ptwo, 0, 1);
	mat_src_t a22 = submatrix_C_ltwh(A, Arowstride, ptwo, ptwo, 1, 1);

	mat_src_t b11 = submatrix_C_ltwh(B, Browstride, ptwo, ptwo, 0, 0);
	mat_src_t b12 = submatrix_C_ltwh(B, Browstride, ptwo, ptwo, 1, 0);
	mat_src_t b21 = submatrix_C_ltwh(B, Browstride, ptwo, ptwo, 0, 1);
	mat_src_t b22 = submatrix_C_ltwh(B, Browstride, ptwo, ptwo, 1, 1);

	mat_t c11 = submatrix_ltwh(C, Crowstride, ptwo, ptwo, 0, 0);
	mat_t c12 = submatrix_ltwh(C, Crowstride, ptwo, ptwo, 1, 0);
	mat_t c21 = submatrix_ltwh(C, Crowstride, ptwo, ptwo, 0, 1);
	mat_t c22 = submatrix_ltwh(C, Crowstride, ptwo, ptwo, 1, 1);

	mat_t p1 = submatrix_ltwh(T, Trowstride, ptwo, ptwo, 0, 0);
	mat_t p2 = submatrix_ltwh(T, Trowstride, ptwo, ptwo, 1, 0);
	mat_t p3 = submatrix_ltwh(T, Trowstride, ptwo, ptwo, 0, 1);
	mat_t p4 = submatrix_ltwh(T, Trowstride, ptwo, ptwo, 1, 1);

	// NOTE: ADD takes (h,w), not (w,h)!
#define ADD(h_, w_,     c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_add(w_, h_, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, d_))
#define MUL(k_, m_, n_, c_, cr_, a_, ar_, b_, br_, d_) job_dep_list(1, matj_mul_squarify(k_, m_, n_, c_, cr_ ## rowstride, a_, ar_ ## rowstride, b_, br_ ## rowstride, basew, baseh, minblocksize, minjobsize, squarefunc, d_))

	struct job_dep_list_t j1, j2, j3, j4, j5, j6, j7, j8, j9, ja, jb, jc;

	j1 = MUL(  ptwo,   ptwo,   ptwo, c11, C, a11, A, b11, B, job_dep_merge(1, &gdeps));
	j2 = MUL(  ptwo, M-ptwo,   ptwo, p1,  T, a12, A, b21, B, job_dep_merge(1, &gdeps));
	j3 = ADD(  ptwo,           ptwo, c11, C, c11, C, p1,  T, job_dep_merge(2, &j1, &j2));

	j4 = MUL(  ptwo,   ptwo, N-ptwo, c12, C, a11, A, b12, B, job_dep_merge(1, &gdeps));
	j5 = MUL(  ptwo, M-ptwo, N-ptwo, p2,  T, a12, A, b22, B, job_dep_merge(1, &gdeps));
	j6 = ADD(  ptwo,         N-ptwo, c12, C, c12, C, p2,  T, job_dep_merge(2, &j4, &j5));

	j7 = MUL(K-ptwo,   ptwo,   ptwo, c21, C, a21, A, b11, B, job_dep_merge(1, &gdeps));
	j8 = MUL(K-ptwo, M-ptwo,   ptwo, p3,  T, a22, A, b21, B, job_dep_merge(1, &gdeps));
	j9 = ADD(K-ptwo,           ptwo, c21, C, c21, C, p3,  T, job_dep_merge(2, &j7, &j8));

	ja = MUL(K-ptwo,   ptwo, N-ptwo, c22, C, a21, A, b12, B, job_dep_merge(1, &gdeps));
	jb = MUL(K-ptwo, M-ptwo, N-ptwo, p4,  T, a22, A, b22, B, job_dep_merge(1, &gdeps));
	jc = ADD(K-ptwo,         N-ptwo, c22, C, c22, C, p4,  T, job_dep_merge(2, &ja, &jb));

#undef ADD
#undef MUL

	job_dep_list_free(gdeps);

	struct job_dep_list_t jobs = job_dep_list(0);
	job_dep_list_append_listp(&jobs, &j3);
	job_dep_list_append_listp(&jobs, &j6);
	job_dep_list_append_listp(&jobs, &j9);
	job_dep_list_append_listp(&jobs, &jc);

	job_dep_list_free(j1); job_dep_list_free(j2); job_dep_list_free(j3); job_dep_list_free(j4);
	job_dep_list_free(j5); job_dep_list_free(j6); job_dep_list_free(j7); job_dep_list_free(j8);
	job_dep_list_free(j9); job_dep_list_free(ja); job_dep_list_free(jb); job_dep_list_free(jc);

	struct jdata_squarify *jdata = malloc(sizeof(struct jdata_squarify));
	jdata->K = K; jdata->M = M; jdata->N = N;
	jdata->T = T;
#ifdef DEBUG
	jdata->C = C; jdata->Crowstride = Crowstride;
#endif

	return job_submit(jobs, squarify_cleanup_j, jdata);
}

// vim: set sw=4 ts=4 noet:
