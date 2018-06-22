#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <alloca.h>


#define __all__ __host__ __device__


#define USE_32

#define PREFIX(name) gpu_ ## name


#ifdef __CUDA_ARCH__
#define UNIMPLEMENTED do { assert(false); } while (0)
#else
#define UNIMPLEMENTED do { abort(); } while (0)
#endif

#ifdef USE_32
typedef uint32_t PREFIX(mp_limb_t);
typedef int32_t PREFIX(mp_limb_signed_t);
typedef int32_t PREFIX(mp_size_t);
#else
typedef unsigned long int PREFIX(mp_limb_t);
typedef long int PREFIX(mp_limb_signed_t);
typedef long int PREFIX(mp_size_t);
#endif

typedef PREFIX(mp_limb_t) *PREFIX(mp_ptr);
typedef const PREFIX(mp_limb_t) *PREFIX(mp_srcptr);


#define __GMP_DECLSPEC

#define ASSERT(expr) do {} while (0)
#define ASSERT_LIMB(limb)       do {} while (0)
#define ASSERT_MPN(ptr, size)   do {} while (0)
#define ASSERT_CARRY(expr)     (expr)
#define ASSERT_NOCARRY(expr)   (expr)

#ifdef __CUDA_ARCH__
#define LIKELY(cond) (cond)
#define UNLIKELY(cond) (cond)
#else
#define LIKELY(cond)    __builtin_expect ((cond) != 0, 1)
#define UNLIKELY(cond)  __builtin_expect ((cond) != 0, 0)
#endif

#define ABOVE_THRESHOLD(size,thresh)                                 \
  ((__builtin_constant_p (thresh) && (thresh) == 0)                  \
   || (!(__builtin_constant_p (thresh) && (thresh) == MP_SIZE_T_MAX) \
       && (size) >= (thresh)))

#define BELOW_THRESHOLD(size,thresh)  (! ABOVE_THRESHOLD (size, thresh))

#define MAX(h,i) ((h) > (i) ? (h) : (i))

#ifdef USE_32
#define GPU_SIZEOF_MP_LIMB_T 4
#else
#define GPU_SIZEOF_MP_LIMB_T 8
#endif

#define GPU_GMP_LIMB_BYTES  GPU_SIZEOF_MP_LIMB_T
#define GPU_GMP_LIMB_BITS  (8 * GPU_SIZEOF_MP_LIMB_T)
#define GPU_GMP_NAIL_BITS                      0

#define GPU_GMP_NUMB_BITS     (GPU_GMP_LIMB_BITS - GPU_GMP_NAIL_BITS)
#define GPU_GMP_NUMB_MASK     ((~ __GMP_CAST (mp_limb_t, 0)) >> GPU_GMP_NAIL_BITS)
#define GPU_GMP_NUMB_MAX      GPU_GMP_NUMB_MASK
#define GPU_GMP_NAIL_MASK     (~ GPU_GMP_NUMB_MASK)

#define CNST_LIMB(C) ((mp_limb_t) C##L)


#define TMP_SDECL
#define TMP_DECL                struct tmp_reentrant_t *__tmp_marker
#define TMP_SMARK
#define TMP_MARK                __tmp_marker = 0
#define TMP_SALLOC(n)                alloca(n)
// #define TMP_BALLOC(n)                __gmp_tmp_reentrant_alloc (&__tmp_marker, n)
#define TMP_BALLOC(n) (abort(), NULL)
/* The peculiar stack allocation limit here is chosen for efficient asm.  */
#define TMP_ALLOC(n)                                                        \
  (LIKELY ((n) <= 0x7f00) ? TMP_SALLOC(n) : TMP_BALLOC(n))
#define TMP_SFREE
#define TMP_FREE do { (void)__tmp_marker; } while (0)
/* #define TMP_FREE                                                        \
  do {                                                                        \
    if (UNLIKELY (__tmp_marker != 0))                                        \
      __gmp_tmp_reentrant_free (__tmp_marker);                                \
  } while (0) */

#define TMP_ALLOC_TYPE(n,type)  ((type *) TMP_ALLOC ((n) * sizeof (type)))
#define TMP_SALLOC_TYPE(n,type) ((type *) TMP_SALLOC ((n) * sizeof (type)))
#define TMP_ALLOC_LIMBS(n)     TMP_ALLOC_TYPE(n,mp_limb_t)
#define TMP_SALLOC_LIMBS(n)     TMP_SALLOC_TYPE(n,mp_limb_t)


/* __GMP_CAST allows us to use static_cast in C++, so our macros are clean
   to "g++ -Wold-style-cast".

   Casts in "extern inline" code within an extern "C" block don't induce
   these warnings, so __GMP_CAST only needs to be used on documented
   macros.  */

#ifdef __cplusplus
#define __GMP_CAST(type, expr)  (static_cast<type> (expr))
#else
#define __GMP_CAST(type, expr)  ((type) (expr))
#endif

#if defined (__cplusplus)
#define __GMP_NOTHROW  throw ()
#else
#define __GMP_NOTHROW
#endif

#define __GMPN_COPY_REST(dst, src, size, start)                 \
  do {                                                          \
    PREFIX(mp_size_t) __gmp_j;                                          \
    /* ASSERT ((size) >= 0); */                                 \
    /* ASSERT ((start) >= 0); */                                \
    /* ASSERT ((start) <= (size)); */                           \
    /* ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, size)); */     \
    /* __GMP_CRAY_Pragma ("_CRI ivdep"); */                     \
    for (__gmp_j = (start); __gmp_j < (size); __gmp_j++)        \
      (dst)[__gmp_j] = (src)[__gmp_j];                          \
  } while (0)

#define __GMPN_COPY_REST_STRIDED(dst, dststr, src, size, srcstr, start) \
  do {                                                          \
    PREFIX(mp_size_t) __gmp_j;                                          \
    for (__gmp_j = (start); __gmp_j < (size); __gmp_j++)        \
      (dst)[__gmp_j * (dststr)] = (src)[__gmp_j * (srcstr)];    \
  } while (0)

#define MPN_COPY_INCR(dst, src, n)					\
  do {									\
    ASSERT ((n) >= 0);							\
    ASSERT (MPN_SAME_OR_INCR_P (dst, src, n));				\
    if ((n) != 0)							\
      {									\
	PREFIX(mp_size_t) __n = (n) - 1;				\
	PREFIX(mp_ptr) __dst = (dst);					\
	PREFIX(mp_srcptr) __src = (src);				\
	PREFIX(mp_limb_t) __x;						\
	__x = *__src++;							\
	if (__n != 0)							\
	  {								\
	    do								\
	      {								\
		*__dst++ = __x;						\
		__x = *__src++;						\
	      }								\
	    while (--__n);						\
	  }								\
	*__dst++ = __x;							\
      }									\
  } while (0)

#define MPN_COPY_INCR_STRIDED(dst, ds, src, ss, n)			\
  do {									\
    if ((n) != 0)							\
      {									\
	PREFIX(mp_size_t) __n = (n) - 1;				\
	PREFIX(mp_ptr) __dst = (dst);					\
	PREFIX(mp_srcptr) __src = (src);				\
	PREFIX(mp_limb_t) __x;						\
	__x = *__src; __src += ss;					\
	if (__n != 0)							\
	  {								\
	    do								\
	      {								\
		*__dst = __x; __dst += ds;				\
		__x = *__src; __src += ss;				\
	      }								\
	    while (--__n);						\
	  }								\
	*__dst = __x; __dst += ds;					\
      }									\
  } while (0)

#define MPN_COPY(d,s,n)							\
  do {									\
    ASSERT (MPN_SAME_OR_SEPARATE_P (d, s, n));				\
    MPN_COPY_INCR (d, s, n);						\
  } while (0)

#define MPN_COPY_STRIDED(d, ds, s, ss, n)				\
  do {									\
    MPN_COPY_INCR_STRIDED (d, ds, s, ss, n);				\
  } while (0)


#define MP_PTR_SWAP(x, y)						\
  do {									\
    PREFIX(mp_ptr) __mp_ptr_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_ptr_swap__tmp;						\
  } while (0)
#define MP_SRCPTR_SWAP(x, y)						\
  do {									\
    PREFIX(mp_srcptr) __mp_srcptr_swap__tmp = (x);			\
    (x) = (y);								\
    (y) = __mp_srcptr_swap__tmp;					\
  } while (0)

#define MPN_PTR_SWAP(xp,xs, yp,ys)					\
  do {									\
    MP_PTR_SWAP (xp, yp);						\
    MP_SIZE_T_SWAP (xs, ys);						\
  } while(0)
#define MPN_SRCPTR_SWAP(xp,xs, yp,ys)					\
  do {									\
    MP_SRCPTR_SWAP (xp, yp);						\
    MP_SIZE_T_SWAP (xs, ys);						\
  } while(0)


#if 0
#define mpn_incr_u(p,incr)						\
  do {									\
    mp_limb_t __x;							\
    mp_ptr __p = (p);							\
    if (__builtin_constant_p (incr) && (incr) == 1)			\
      {									\
	while (++(*(__p++)) == 0)					\
	  ;								\
      }									\
    else								\
      {									\
	__x = *__p + (incr);						\
	*__p = __x;							\
	if (__x < (incr))						\
	  while (++(*(++__p)) == 0)					\
	    ;								\
      }									\
  } while (0)
#endif

#define mpn_incr_u_1(p)							\
  do {									\
    PREFIX(mp_ptr) __p = (p);						\
    while (++(*(__p++)) == 0)						\
      ;									\
  } while (0)

#define mpn_incr_u_1_strided(p, ps)							\
  do {									\
    PREFIX(mp_ptr) __p = (p);						\
    while (++(*__p) == 0)						\
      __p += ps;									\
  } while (0)

#define mpn_incr_u_var(p,incr)						\
  do {									\
    PREFIX(mp_limb_t) __x;						\
    PREFIX(mp_ptr) __p = (p);						\
    __x = *__p + (incr);						\
    *__p = __x;								\
    if (__x < (incr))							\
      while (++(*(++__p)) == 0)						\
        ;								\
  } while (0)

#define mpn_incr_u_var_strided(p, ps, incr)						\
  do {									\
    PREFIX(mp_limb_t) __x;						\
    PREFIX(mp_ptr) __p = (p);						\
    __x = *__p + (incr);						\
    *__p = __x;								\
    if (__x < (incr))							\
      while (++(*(__p += ps)) == 0)						\
        ;								\
  } while (0)

#if 0
#define mpn_decr_u(p,incr)						\
  do {									\
    mp_limb_t __x;							\
    mp_ptr __p = (p);							\
    if (__builtin_constant_p (incr) && (incr) == 1)			\
      {									\
	while ((*(__p++))-- == 0)					\
	  ;								\
      }									\
    else								\
      {									\
	__x = *__p;							\
	*__p = __x - (incr);						\
	if (__x < (incr))						\
	  while ((*(++__p))-- == 0)					\
	    ;								\
      }									\
  } while (0)
#endif

#define mpn_decr_u_1(p)							\
  do {									\
    PREFIX(mp_ptr) __p = (p);						\
    while ((*(__p++))-- == 0)						\
      ;									\
  } while (0)

#define mpn_decr_u_1_strided(p, ps)							\
  do {									\
    PREFIX(mp_ptr) __p = (p);						\
    while ((*__p)-- == 0)						\
      __p += ps;									\
  } while (0)

#define mpn_decr_u_var(p,incr)						\
  do {									\
    PREFIX(mp_limb_t) __x;						\
    PREFIX(mp_ptr) __p = (p);						\
    __x = *__p;								\
    *__p = __x - (incr);						\
    if (__x < (incr))							\
      while ((*(++__p))-- == 0)						\
	;								\
  } while (0)

#define mpn_decr_u_var_strided(p, ps, incr)						\
  do {									\
    PREFIX(mp_limb_t) __x;						\
    PREFIX(mp_ptr) __p = (p);						\
    __x = *__p;								\
    *__p = __x - (incr);						\
    if (__x < (incr))							\
      while ((*(__p += ps))-- == 0)						\
	;								\
  } while (0)

#if 0
#define MPN_INCR_U(ptr, size, n)   mpn_incr_u (ptr, n)
#define MPN_DECR_U(ptr, size, n)   mpn_decr_u (ptr, n)
#endif

#define MPN_INCR_U_1(ptr, size) mpn_incr_u_1 (ptr)
#define MPN_DECR_U_1(ptr, size) mpn_decr_u_1 (ptr)
#define MPN_INCR_U_VAR(ptr, size, n) mpn_incr_u_var (ptr, n)
#define MPN_DECR_U_VAR(ptr, size, n) mpn_decr_u_var (ptr, n)

#define MPN_INCR_U_1_STRIDED(ptr, size, stride) mpn_incr_u_1_strided (ptr, stride)
#define MPN_DECR_U_1_STRIDED(ptr, size, stride) mpn_decr_u_1_strided (ptr, stride)
#define MPN_INCR_U_VAR_STRIDED(ptr, size, stride, n) mpn_incr_u_var_strided (ptr, stride, n)
#define MPN_DECR_U_VAR_STRIDED(ptr, size, stride, n) mpn_decr_u_var_strided (ptr, stride, n)


#define MPN_FILL(dst, n, f)						\
  do {									\
    PREFIX(mp_ptr) __dst = (dst);					\
    PREFIX(mp_size_t) __n = (n);					\
    ASSERT (__n > 0);							\
    do									\
      *__dst++ = (f);							\
    while (--__n);							\
  } while (0)

#define MPN_FILL_STRIDED(dst, n, s, f)						\
  do {									\
    PREFIX(mp_ptr) __dst = (dst);					\
    PREFIX(mp_size_t) __n = (n);					\
    ASSERT (__n > 0);							\
    do {									\
      *__dst = (f);							\
      __dst += s;                               \
	} while (--__n);							\
  } while (0)

#define MPN_ZERO(dst, n)						\
  do {									\
    ASSERT ((n) >= 0);							\
    if ((n) != 0)							\
      MPN_FILL (dst, n, CNST_LIMB (0));					\
  } while (0)

#define MPN_ZERO_STRIDED(dst, n, s)						\
  do {									\
    ASSERT ((n) >= 0);							\
    if ((n) != 0)							\
      MPN_FILL_STRIDED (dst, n, s, CNST_LIMB (0));					\
  } while (0)


// We haven't implemented toom33, so just use toom22 for longer
// #define MUL_TOOM22_THRESHOLD             30
#define MUL_TOOM22_THRESHOLD            100

#define MUL_TOOM33_THRESHOLD            100
#define MUL_TOOM44_THRESHOLD            300
#define MUL_TOOM6H_THRESHOLD            350
#define SQR_TOOM6_THRESHOLD MUL_TOOM6H_THRESHOLD
#define MUL_TOOM8H_THRESHOLD            450
#define SQR_TOOM8_THRESHOLD MUL_TOOM8H_THRESHOLD
#define MUL_TOOM32_TO_TOOM43_THRESHOLD  100
#define MUL_TOOM32_TO_TOOM53_THRESHOLD  110
#define MUL_TOOM42_TO_TOOM53_THRESHOLD  100
#define MUL_TOOM42_TO_TOOM63_THRESHOLD  110
#define MUL_TOOM43_TO_TOOM54_THRESHOLD  150

/* MUL_TOOM22_THRESHOLD_LIMIT is the maximum for MUL_TOOM22_THRESHOLD.  In a
   normal build MUL_TOOM22_THRESHOLD is a constant and we use that.  In a fat
   binary or tune program build MUL_TOOM22_THRESHOLD is a variable and a
   separate hard limit will have been defined.  Similarly for TOOM3.  */
#define MUL_TOOM22_THRESHOLD_LIMIT  MUL_TOOM22_THRESHOLD
#define MUL_TOOM33_THRESHOLD_LIMIT  MUL_TOOM33_THRESHOLD
#define MULLO_BASECASE_THRESHOLD_LIMIT  MULLO_BASECASE_THRESHOLD
#define SQRLO_BASECASE_THRESHOLD_LIMIT  SQRLO_BASECASE_THRESHOLD
#define SQRLO_DC_THRESHOLD_LIMIT  SQRLO_DC_THRESHOLD

/* SQR_BASECASE_THRESHOLD is where mpn_sqr_basecase should take over from
   mpn_mul_basecase.  Default is to use mpn_sqr_basecase from 0.  (Note that we
   certainly always want it if there's a native assembler mpn_sqr_basecase.)

   If it turns out that mpn_toom2_sqr becomes faster than mpn_mul_basecase
   before mpn_sqr_basecase does, then SQR_BASECASE_THRESHOLD is the toom2
   threshold and SQR_TOOM2_THRESHOLD is 0.  This oddity arises more or less
   because SQR_TOOM2_THRESHOLD represents the size up to which mpn_sqr_basecase
   should be used, and that may be never.  */

#define SQR_BASECASE_THRESHOLD            0  /* never use mpn_mul_basecase */
#define SQR_TOOM2_THRESHOLD              50
#define SQR_TOOM3_THRESHOLD             120
#define SQR_TOOM4_THRESHOLD             400

/* See comments above about MUL_TOOM33_THRESHOLD_LIMIT.  */
#define SQR_TOOM3_THRESHOLD_LIMIT  SQR_TOOM3_THRESHOLD

/* First k to use for an FFT modF multiply.  A modF FFT is an order
   log(2^k)/log(2^(k-1)) algorithm, so k=3 is merely 1.5 like karatsuba,
   whereas k=4 is 1.33 which is faster than toom3 at 1.485.    */
#define FFT_FIRST_K  4

/* Threshold at which FFT should be used to do a modF NxN -> N multiply. */
#define MUL_FFT_MODF_THRESHOLD   (MUL_TOOM33_THRESHOLD * 3)
#define SQR_FFT_MODF_THRESHOLD   (SQR_TOOM3_THRESHOLD * 3)

/* Threshold at which FFT should be used to do an NxN -> 2N multiply.  This
   will be a size where FFT is using k=7 or k=8, since an FFT-k used for an
   NxN->2N multiply and not recursing into itself is an order
   log(2^k)/log(2^(k-2)) algorithm, so it'll be at least k=7 at 1.39 which
   is the first better than toom3.  */
#define MUL_FFT_THRESHOLD   (MUL_FFT_MODF_THRESHOLD * 10)
#define SQR_FFT_THRESHOLD   (SQR_FFT_MODF_THRESHOLD * 10)

/* toom22/toom2: Scratch need is 2*(an + k), k is the recursion depth.
   k is ths smallest k such that
     ceil(an/2^k) < MUL_TOOM22_THRESHOLD.
   which implies that
     k = bitsize of floor ((an-1)/(MUL_TOOM22_THRESHOLD-1))
       = 1 + floor (log_2 (floor ((an-1)/(MUL_TOOM22_THRESHOLD-1))))
*/
#define mpn_toom22_mul_itch(an, bn) \
  (2 * ((an) + GPU_GMP_NUMB_BITS))
#define mpn_toom2_sqr_itch(an) \
  (2 * ((an) + GPU_GMP_NUMB_BITS))

/* toom33/toom3: Scratch need is 5an/2 + 10k, k is the recursion depth.
   We use 3an + C, so that we can use a smaller constant.
 */
#define mpn_toom33_mul_itch(an, bn) \
  (3 * (an) + GPU_GMP_NUMB_BITS)
#define mpn_toom3_sqr_itch(an) \
  (3 * (an) + GPU_GMP_NUMB_BITS)

/* toom33/toom3: Scratch need is 8an/3 + 13k, k is the recursion depth.
   We use 3an + C, so that we can use a smaller constant.
 */
#define mpn_toom44_mul_itch(an, bn) \
  (3 * (an) + GPU_GMP_NUMB_BITS)
#define mpn_toom4_sqr_itch(an) \
  (3 * (an) + GPU_GMP_NUMB_BITS)

#define mpn_toom6_sqr_itch(n)                                                \
  (((n) - SQR_TOOM6_THRESHOLD)*2 +                                        \
   MAX(SQR_TOOM6_THRESHOLD*2 + GPU_GMP_NUMB_BITS*6,                                \
       mpn_toom4_sqr_itch(SQR_TOOM6_THRESHOLD)))

#define MUL_TOOM6H_MIN                                                        \
  ((MUL_TOOM6H_THRESHOLD > MUL_TOOM44_THRESHOLD) ?                        \
    MUL_TOOM6H_THRESHOLD : MUL_TOOM44_THRESHOLD)
#define mpn_toom6_mul_n_itch(n)                                                \
  (((n) - MUL_TOOM6H_MIN)*2 +                                                \
   MAX(MUL_TOOM6H_MIN*2 + GPU_GMP_NUMB_BITS*6,                                \
       mpn_toom44_mul_itch(MUL_TOOM6H_MIN,MUL_TOOM6H_MIN)))

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom6h_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn) {
  PREFIX(mp_size_t) estimatedN;
  estimatedN = (an + bn) / (size_t) 10 + 1;
  return mpn_toom6_mul_n_itch (estimatedN * 6);
}

#define mpn_toom8_sqr_itch(n)                                                \
  ((((n)*15)>>3) - ((SQR_TOOM8_THRESHOLD*15)>>3) +                        \
   MAX(((SQR_TOOM8_THRESHOLD*15)>>3) + GPU_GMP_NUMB_BITS*6,                        \
       PREFIX(mpn_toom6_sqr_itch)(SQR_TOOM8_THRESHOLD)))

#define MUL_TOOM8H_MIN                                                        \
  ((MUL_TOOM8H_THRESHOLD > MUL_TOOM6H_MIN) ?                                \
    MUL_TOOM8H_THRESHOLD : MUL_TOOM6H_MIN)
#define mpn_toom8_mul_n_itch(n)                                                \
  ((((n)*15)>>3) - ((MUL_TOOM8H_MIN*15)>>3) +                                \
   MAX(((MUL_TOOM8H_MIN*15)>>3) + GPU_GMP_NUMB_BITS*6,                        \
       mpn_toom6_mul_n_itch(MUL_TOOM8H_MIN)))

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom8h_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn) {
  PREFIX(mp_size_t) estimatedN;
  estimatedN = (an + bn) / (size_t) 14 + 1;
  return mpn_toom8_mul_n_itch (estimatedN * 8);
}

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom32_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn)
{
  PREFIX(mp_size_t) n = 1 + (2 * an >= 3 * bn ? (an - 1) / (size_t) 3 : (bn - 1) >> 1);
  PREFIX(mp_size_t) itch = 2 * n + 1;

  return itch;
}

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom42_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn)
{
  PREFIX(mp_size_t) n = an >= 2 * bn ? (an + 3) >> 2 : (bn + 1) >> 1;
  return 6 * n + 3;
}

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom43_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn)
{
  PREFIX(mp_size_t) n = 1 + (3 * an >= 4 * bn ? (an - 1) >> 2 : (bn - 1) / (size_t) 3);

  return 6*n + 4;
}

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom52_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn)
{
  PREFIX(mp_size_t) n = 1 + (2 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) >> 1);
  return 6*n + 4;
}

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom53_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn)
{
  PREFIX(mp_size_t) n = 1 + (3 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) / (size_t) 3);
  return 10 * n + 10;
}

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom62_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn)
{
  PREFIX(mp_size_t) n = 1 + (an >= 3 * bn ? (an - 1) / (size_t) 6 : (bn - 1) >> 1);
  return 10 * n + 10;
}

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom63_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn)
{
  PREFIX(mp_size_t) n = 1 + (an >= 2 * bn ? (an - 1) / (size_t) 6 : (bn - 1) / (size_t) 3);
  return 9 * n + 3;
}

static inline PREFIX(mp_size_t)
PREFIX(mpn_toom54_mul_itch) (PREFIX(mp_size_t) an, PREFIX(mp_size_t) bn)
{
  PREFIX(mp_size_t) n = 1 + (4 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) / (size_t) 4);
  return 9 * n + 3;
}

/* let S(n) = space required for input size n,
   then S(n) = 3 floor(n/2) + 1 + S(floor(n/2)).   */
#define mpn_toom42_mulmid_itch(n) \
  (3 * (n) + GPU_GMP_NUMB_BITS)


#ifdef USE_32
#define MP_SIZE_T_MAX      INT32_MAX
#define MP_SIZE_T_MIN      INT32_MIN
#else
#define MP_SIZE_T_MAX      LONG_MAX
#define MP_SIZE_T_MIN      LONG_MIN
#endif


#ifdef USE_32
/*# ifdef __CUDA_ARCH__
# define umul_ppmm(w1, w0, u, v) \
  do { \
    uint32_t __u = (u), __v = (v); \
    (w0) = (uint32_t)__u * (uint32_t)__v; \
    (w1) = __umulhi(__u, __v); \
  } while (0)
# else*/
# define umul_ppmm(w1, w0, u, v) \
  do { \
    uint64_t __r = (uint64_t)(u) * (uint64_t)(v); \
    (w0) = __r; \
    (w1) = __r >> 32; \
  } while (0)
// # endif
#else
typedef unsigned long long int UDItype;

#define umul_ppmm(w1, w0, u, v) \
  __asm__ ("mulq %3"							\
	   : "=a" (w0), "=d" (w1)					\
	   : "%0" ((UDItype)(u)), "rm" ((UDItype)(v)))
#endif

#define SUBC_LIMB(cout, w, x, y)					\
  do {									\
    mp_limb_t  __x = (x);						\
    mp_limb_t  __y = (y);						\
    mp_limb_t  __w = __x - __y;						\
    (w) = __w;								\
    (cout) = __w > __x;							\
  } while (0)

#define count_trailing_zeros_gcc_ctz(count,x)	\
  do {						\
    ASSERT ((x) != 0);				\
    (count) = __builtin_ctzl (x);		\
  } while (0)

#define count_trailing_zeros(count, x)  count_trailing_zeros_gcc_ctz(count, x)

#define binvert_limb_table  __gmp_binvert_limb_table
__GMP_DECLSPEC extern const unsigned char  binvert_limb_table[128];

#define binvert_limb(inv,n)						\
  do {									\
    mp_limb_t  __n = (n);						\
    mp_limb_t  __inv;							\
    ASSERT ((__n & 1) == 1);						\
									\
    __inv = binvert_limb_table[(__n/2) & 0x7F]; /*  8 */		\
    if (GPU_GMP_NUMB_BITS > 8)   __inv = 2 * __inv - __inv * __inv * __n;	\
    if (GPU_GMP_NUMB_BITS > 16)  __inv = 2 * __inv - __inv * __inv * __n;	\
    if (GPU_GMP_NUMB_BITS > 32)  __inv = 2 * __inv - __inv * __inv * __n;	\
									\
    if (GPU_GMP_NUMB_BITS > 64)						\
      {									\
	int  __invbits = 64;						\
	do {								\
	  __inv = 2 * __inv - __inv * __inv * __n;			\
	  __invbits *= 2;						\
	} while (__invbits < GPU_GMP_NUMB_BITS);				\
      }									\
									\
    ASSERT ((__inv * __n & GMP_NUMB_MASK) == 1);			\
    (inv) = __inv & GMP_NUMB_MASK;					\
  } while (0)


enum PREFIX(toom6_flags) { PREFIX(toom6_all_pos) = 0, PREFIX(toom6_vm1_neg) = 1, PREFIX(toom6_vm2_neg) = 2 };
enum PREFIX(toom7_flags) { PREFIX(toom7_w1_neg) = 1, PREFIX(toom7_w3_neg) = 2 };

// Note: ensure un >= vn for mpn_mul and mpn_mul_basecase.
__all__ PREFIX(mp_limb_t) PREFIX(mpn_mul) (PREFIX(mp_ptr) prodp, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) un, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) vn);
__all__ void PREFIX(mpn_mul_basecase) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) un, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) vn);
__all__ void PREFIX(mpn_mul_n) (PREFIX(mp_ptr) p, PREFIX(mp_srcptr) a, PREFIX(mp_srcptr) b, PREFIX(mp_size_t) n);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_mul_1) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) n, PREFIX(mp_limb_t) vl);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_addmul_1) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) n, PREFIX(mp_limb_t) vl);
PREFIX(mp_limb_t) PREFIX(mpn_submul_1) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) n, PREFIX(mp_limb_t) vl);
__all__ void PREFIX(mpn_toom22_mul) (PREFIX(mp_ptr) pp, PREFIX(mp_srcptr) ap, PREFIX(mp_size_t) an, PREFIX(mp_srcptr) bp, PREFIX(mp_size_t) bn, PREFIX(mp_ptr) scratch);
void PREFIX(mpn_toom32_mul) (PREFIX(mp_ptr) pp, PREFIX(mp_srcptr) ap, PREFIX(mp_size_t) an, PREFIX(mp_srcptr) bp, PREFIX(mp_size_t) bn, PREFIX(mp_ptr) scratch);
void PREFIX(mpn_toom42_mul) (PREFIX(mp_ptr) pp, PREFIX(mp_srcptr) ap, PREFIX(mp_size_t) an, PREFIX(mp_srcptr) bp, PREFIX(mp_size_t) bn, PREFIX(mp_ptr) scratch);
void PREFIX(mpn_toom52_mul) (PREFIX(mp_ptr) pp, PREFIX(mp_srcptr) ap, PREFIX(mp_size_t) an, PREFIX(mp_srcptr) bp, PREFIX(mp_size_t) bn, PREFIX(mp_ptr) scratch);
void PREFIX(mpn_toom62_mul) (PREFIX(mp_ptr) pp, PREFIX(mp_srcptr) ap, PREFIX(mp_size_t) an, PREFIX(mp_srcptr) bp, PREFIX(mp_size_t) bn, PREFIX(mp_ptr) scratch);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_add) (PREFIX(mp_ptr) __gmp_wp, PREFIX(mp_srcptr) __gmp_xp, PREFIX(mp_size_t) __gmp_xsize, PREFIX(mp_srcptr) __gmp_yp, PREFIX(mp_size_t) __gmp_ysize);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_add_1) (PREFIX(mp_ptr) __gmp_dst, PREFIX(mp_srcptr) __gmp_src, PREFIX(mp_size_t) __gmp_size, PREFIX(mp_limb_t) __gmp_n) __GMP_NOTHROW;
__all__ PREFIX(mp_limb_t) PREFIX(mpn_sub) (PREFIX(mp_ptr) __gmp_wp, PREFIX(mp_srcptr) __gmp_xp, PREFIX(mp_size_t) __gmp_xsize, PREFIX(mp_srcptr) __gmp_yp, PREFIX(mp_size_t) __gmp_ysize);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_sub_1) (PREFIX(mp_ptr) __gmp_dst, PREFIX(mp_srcptr) __gmp_src, PREFIX(mp_size_t) __gmp_size, PREFIX(mp_limb_t) __gmp_n) __GMP_NOTHROW;
__all__ PREFIX(mp_limb_t) PREFIX(mpn_neg) (PREFIX(mp_ptr) __gmp_rp, PREFIX(mp_srcptr) __gmp_up, PREFIX(mp_size_t) __gmp_n);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_add_n) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) n);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_sub_n) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) n);
__all__ int PREFIX(mpn_cmp) (PREFIX(mp_srcptr) __gmp_xp, PREFIX(mp_srcptr) __gmp_yp, PREFIX(mp_size_t) __gmp_size) __GMP_NOTHROW;
__all__ int PREFIX(mpn_zero_p) (PREFIX(mp_srcptr) __gmp_p, PREFIX(mp_size_t) __gmp_n) __GMP_NOTHROW;
__all__ void PREFIX(mpn_com) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) n);

__all__ PREFIX(mp_limb_t) PREFIX(mpn_mul_strided) (PREFIX(mp_ptr) rp, PREFIX(mp_size_t) rs, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) un, PREFIX(mp_size_t) us, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) vn, PREFIX(mp_size_t) vs);
__all__ void PREFIX(mpn_mul_basecase_strided) (PREFIX(mp_ptr) rp, PREFIX(mp_size_t) rs, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) un, PREFIX(mp_size_t) us, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) vn, PREFIX(mp_size_t) vs);
__all__ void PREFIX(mpn_mul_n_strided) (PREFIX(mp_ptr) p, PREFIX(mp_size_t) ps, PREFIX(mp_srcptr) a, PREFIX(mp_size_t) as, PREFIX(mp_srcptr) b, PREFIX(mp_size_t) bs, PREFIX(mp_size_t) n);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_mul_1_strided) (PREFIX(mp_ptr) rp, PREFIX(mp_size_t) rs, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) n, PREFIX(mp_size_t) us, PREFIX(mp_limb_t) vl);
__all__ void PREFIX(mpn_toom22_mul_strided) (PREFIX(mp_ptr) pp, PREFIX(mp_size_t) ps, PREFIX(mp_srcptr) ap, PREFIX(mp_size_t) an, PREFIX(mp_size_t) as, PREFIX(mp_srcptr) bp, PREFIX(mp_size_t) bn, PREFIX(mp_size_t) bs, PREFIX(mp_ptr) scratch, PREFIX(mp_size_t) ss);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_add_strided) (PREFIX(mp_ptr) wp, PREFIX(mp_size_t) ws, PREFIX(mp_srcptr) xp, PREFIX(mp_size_t) xsize, PREFIX(mp_size_t) xs, PREFIX(mp_srcptr) yp, PREFIX(mp_size_t) ysize, PREFIX(mp_size_t) ys);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_sub_strided) (PREFIX(mp_ptr) wp, PREFIX(mp_size_t) ws, PREFIX(mp_srcptr) xp, PREFIX(mp_size_t) xsize, PREFIX(mp_size_t) xs, PREFIX(mp_srcptr) yp, PREFIX(mp_size_t) ysize, PREFIX(mp_size_t) ys);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_add_n_strided) (PREFIX(mp_ptr) rp, PREFIX(mp_size_t) rs, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) us, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) vs, PREFIX(mp_size_t) n);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_sub_n_strided) (PREFIX(mp_ptr) rp, PREFIX(mp_size_t) rs, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) us, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) vs, PREFIX(mp_size_t) n);
__all__ PREFIX(mp_limb_t) PREFIX(mpn_addmul_1_strided) (PREFIX(mp_ptr) rp, PREFIX(mp_size_t) rs, PREFIX(mp_srcptr) up, PREFIX(mp_size_t) n, PREFIX(mp_size_t) us, PREFIX(mp_limb_t) vl);
__all__ int PREFIX(mpn_cmp_strided) (PREFIX(mp_srcptr) xp, PREFIX(mp_size_t) xs, PREFIX(mp_srcptr) yp, PREFIX(mp_size_t) ys, PREFIX(mp_size_t) size) __GMP_NOTHROW;
__all__ int PREFIX(mpn_zero_p_strided) (PREFIX(mp_srcptr) p, PREFIX(mp_size_t) n, PREFIX(mp_size_t) s) __GMP_NOTHROW;

static inline
PREFIX(mp_limb_t)
PREFIX(mpn_add_nc) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) n, PREFIX(mp_limb_t) ci)
{
  PREFIX(mp_limb_t) co;
  co = PREFIX(mpn_add_n) (rp, up, vp, n);
  co += PREFIX(mpn_add_1) (rp, rp, n, ci);
  return co;
}

static inline PREFIX(mp_limb_t)
PREFIX(mpn_sub_nc) (PREFIX(mp_ptr) rp, PREFIX(mp_srcptr) up, PREFIX(mp_srcptr) vp, PREFIX(mp_size_t) n, PREFIX(mp_limb_t) ci)
{
  PREFIX(mp_limb_t) co;
  co = PREFIX(mpn_sub_n) (rp, up, vp, n);
  co += PREFIX(mpn_sub_1) (rp, rp, n, ci);
  return co;
}

// vim: set sw=2 ts=8 noet:
