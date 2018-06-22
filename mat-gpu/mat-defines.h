#pragma once

#define CAT(a, b) CAT_(a, b)
#define CAT_(a, b) a ## b

#define SPLIT_MATRIX_(cnst_, prefix_, w_, h_, mat_, rs_) \
	cnst_ struct gpu___mpz_t *CAT(prefix_, 11) = submatrix(w_, h_, mat_, rs_, 0, 0); \
	cnst_ struct gpu___mpz_t *CAT(prefix_, 12) = submatrix(w_, h_, mat_, rs_, 1, 0); \
	cnst_ struct gpu___mpz_t *CAT(prefix_, 21) = submatrix(w_, h_, mat_, rs_, 0, 1); \
	cnst_ struct gpu___mpz_t *CAT(prefix_, 22) = submatrix(w_, h_, mat_, rs_, 1, 1);

#define SPLIT_MATRIX(prefix_, w_, h_, mat_, rs_) SPLIT_MATRIX_(, prefix_, w_, h_, mat_, rs_)
#define SPLIT_MATRIX_CONST(prefix_, w_, h_, mat_, rs_) SPLIT_MATRIX_(const, prefix_, w_, h_, mat_, rs_)

#define SPLIT_MATRIX_NUM(prefix_, w_, h_, mat_, rs_, n1_, n2_, n3_, n4_) \
	struct gpu___mpz_t *CAT(prefix_, n1_) = submatrix(w_, h_, mat_, rs_, 0, 0); \
	struct gpu___mpz_t *CAT(prefix_, n2_) = submatrix(w_, h_, mat_, rs_, 1, 0); \
	struct gpu___mpz_t *CAT(prefix_, n3_) = submatrix(w_, h_, mat_, rs_, 0, 1); \
	struct gpu___mpz_t *CAT(prefix_, n4_) = submatrix(w_, h_, mat_, rs_, 1, 1);

#define NAIVE_RECURSIVE_SEQUENCE \
	MUL(c11, C, a11, A, b11, B); \
	MUL(T,   T, a12, A, b21, B); \
	ADD(c11, C, c11, C, T,   T); \
	\
	MUL(c12, C, a11, A, b12, B); \
	MUL(T,   T, a12, A, b22, B); \
	ADD(c12, C, c12, C, T,   T); \
	\
	MUL(c21, C, a21, A, b11, B); \
	MUL(T,   T, a22, A, b21, B); \
	ADD(c21, C, c21, C, T,   T); \
	\
	MUL(c22, C, a21, A, b12, B); \
	MUL(T,   T, a22, A, b22, B); \
	ADD(c22, C, c22, C, T,   T);

#if 0
// Using 8 temporaries for more parallelism
#define STRASSEN_SEQUENCE \
	ADD(p5,  T, a11, A, a22, A); \
	ADD(p6,  T, b11, B, b22, B); \
	MUL(p1,  T, p5,  T, p6,  T); \
	\
	ADD(p7,  T, a21, A, a22, A); \
	MUL(p2,  T, p7,  T, b11, B); \
	\
	SUB(p8,  T, b12, B, b22, B); \
	MUL(p3,  T, a11, A, p8,  T); \
	\
	SUB(p5,  T, b21, B, b11, B); \
	MUL(p4,  T, a22, A, p5,  T); \
	\
	SUB(c11, C, a21, A, a11, A); \
	ADD(c12, C, b11, B, b12, B); \
	MUL(p6,  T, c11, C, c12, C); \
	\
	SUB(c21, C, a12, A, a22, A); \
	ADD(c22, C, b21, B, b22, B); \
	MUL(p7,  T, c21, C, c22, C); \
	\
	ADD(p8,  T, a11, A, a12, A); \
	MUL(p5,  T, p8,  T, b22, B); \
	\
	ADD(c11, C, p1,  T, p4,  T); \
	ADD(c11, C, c11, C, p7,  T); \
	SUB(c11, C, c11, C, p5,  T); \
	\
	ADD(c21, C, p2,  T, p4,  T); \
	\
	ADD(c12, C, p3,  T, p5,  T); \
	\
	ADD(c22, C, p1,  T, p3,  T); \
	SUB(c22, C, c22, C, p2,  T); \
	ADD(c22, C, c22, C, p6,  T);
#else
// Using 3 temporaries for less memory usage
#define STRASSEN_SEQUENCE \
	ADD(p1,  T, a11, A, a22, A); \
	ADD(p2,  T, b11, B, b22, B); \
	MUL(c11, C, p1,  T, p2,  T); \
	\
	SUB(p1,  T, b12, B, b22, B); \
	MUL(c12, C, a11, A, p1,  T); \
	\
	ADD(c22, C, c11, C, c12, C); \
	\
	ADD(p1,  T, a21, A, a22, A); \
	MUL(c21, C, p1,  T, b11, B); \
	\
	SUB(c22, C, c22, C, c21, C); \
	\
	SUB(p2,  T, a12, A, a22, A); \
	ADD(p3,  T, b21, B, b22, B); \
	MUL(p1,  T, p2,  T, p3,  T); \
	\
	ADD(c11, C, c11, C, p1,  T); \
	\
	SUB(p2,  T, a21, A, a11, A); \
	ADD(p3,  T, b11, B, b12, B); \
	MUL(p1,  T, p2,  T, p3,  T); \
	\
	ADD(c22, C, c22, C, p1,  T); \
	\
	SUB(p2,  T, b21, B, b11, B); \
	MUL(p1,  T, a22, A, p2,  T); \
	\
	ADD(c11, C, c11, C, p1,  T); \
	\
	ADD(c21, C, c21, C, p1,  T); \
	\
	ADD(p2,  T, a11, A, a12, A); \
	MUL(p1,  T, p2,  T, b22, B); \
	\
	SUB(c11, C, c11, C, p1,  T); \
	\
	ADD(c12, C, c12, C, p1,  T);
#endif

// #define WINOGRAD_SEQUENCE \
//     ADD(p1,  T, a21, A, a22, A); \
//     SUB(p2,  T, b12, B, b11, B); \
//     MUL(c22, C, p1,  T, p2,  T); \
//     SUB(p1,  T, p1,  T, a11, A); \
//     SUB(p2,  T, b22, B, p2,  T); \
//     MUL(p3,  T, a11, A, b11, B); \
//     MUL(c11, C, a12, A, b21, B); \
//     ADD(c11, C, c11, C, p3,  T); \
//     MUL(p4,  T, p1,  T, p2,  T); \
//     ADD(p3,  T, p3,  T, p4,  T); \
//     SUB(p1,  T, a12, A, p1,  T); \
//     SUB(p2,  T, b21, B, p2,  T); \
//     MUL(c12, C, p1,  T, b22, B); \
//     ADD(c12, C, c12, C, c22, C); \
//     ADD(c12, C, c12, C, p3,  T); \
//     MUL(c21, C, a22, A, p2,  T); \
//     SUB(p1,  T, a11, A, a21, A); \
//     SUB(p2,  T, b22, B, b12, B); \
//     MUL(p4,  T, p1,  T, p2,  T); \
//     ADD(p3,  T, p3,  T, p4,  T); \
//     ADD(c21, C, c21, C, p3,  T); \
//     ADD(c22, C, c22, C, p3,  T);

#define WINOGRAD_SEQUENCE \
	ADD(p1,  T, a21, A, a22, A); \
	SUB(p2,  T, b12, B, b11, B); \
	MUL(c22, C, p1,  T, p2,  T); \
	SUB(p1,  T, p1,  T, a11, A); \
	SUB(p2,  T, b22, B, p2,  T); \
	MUL(p3,  T, a11, A, b11, B); \
	MUL(c11, C, a12, A, b21, B); \
	ADD(c11, C, c11, C, p3,  T); \
	AML(p3,  T, p1,  T, p2,  T); \
	SUB(p1,  T, a12, A, p1,  T); \
	SUB(p2,  T, b21, B, p2,  T); \
	MUL(c12, C, p1,  T, b22, B); \
	ADD(c12, C, c12, C, c22, C); \
	ADD(c12, C, c12, C, p3,  T); \
	MUL(c21, C, a22, A, p2,  T); \
	SUB(p1,  T, a11, A, a21, A); \
	SUB(p2,  T, b22, B, b12, B); \
	AML(p3,  T, p1,  T, p2,  T); \
	ADD(c21, C, c21, C, p3,  T); \
	ADD(c22, C, c22, C, p3,  T);

#define WINOGRAD_ADDMUL_SEQUENCE \
	ADD(p1,  T, a21, A, a22, A); \
	SUB(p2,  T, b12, B, b11, B); \
	MUL(p3,  T, p1,  T, p2,  T); \
	ADD(c12, C, c12, C, p3,  T); \
	ADD(c22, C, c22, C, p3,  T); \
	SUB(p1,  T, p1,  T, a11, A); \
	SUB(p2,  T, b22, B, p2,  T); \
	MUL(p3,  T, a11, A, b11, B); \
	ADD(c11, C, c11, C, p3,  T); \
	AML(p3,  T, p1,  T, p2,  T); \
	AML(c11, C, a12, A, b21, B); \
	SUB(p1,  T, a12, A, p1,  T); \
	SUB(p2,  T, b21, B, p2,  T); \
	AML(c12, C, p1,  T, b22, B); \
	ADD(c12, C, c12, C, p3,  T); \
	AML(c21, C, a22, A, p2,  T); \
	SUB(p1,  T, a11, A, a21, A); \
	SUB(p2,  T, b22, B, b12, B); \
	AML(p3,  T, p1,  T, p2,  T); \
	ADD(c21, C, c21, C, p3,  T); \
	ADD(c22, C, c22, C, p3,  T);


// vim: set sw=4 ts=4 noet:
