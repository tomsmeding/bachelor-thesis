#include "gmp-es.h"
#include "gmp-es-internal-convert.h"

/* The comments with __GMPN_ADD_1 below apply here too.

   The test for FUNCTION returning 0 should predict well.  If it's assumed
   {yp,ysize} will usually have a random number of bits then the high limb
   won't be full and a carry out will occur a good deal less than 50% of the
   time.

   ysize==0 isn't a documented feature, but is used internally in a few
   places.

   Producing cout last stops it using up a register during the main part of
   the calculation, though gcc (as of 3.0) on an "if (mpn_add (...))"
   doesn't seem able to move the true and false legs of the conditional up
   to the two places cout is generated.  */

#define __GMPN_AORS(cout, wp, xp, xsize, yp, ysize, FUNCTION, TEST)     \
  do {                                                                  \
    mp_size_t  __gmp_i;                                                 \
    mp_limb_t  __gmp_x;                                                 \
                                                                        \
    ASSERT ((ysize) >= 0);                                              \
    ASSERT ((xsize) >= (ysize));                                        \
    ASSERT (MPN_SAME_OR_SEPARATE2_P (wp, xsize, xp, xsize));            \
    ASSERT (MPN_SAME_OR_SEPARATE2_P (wp, xsize, yp, ysize));            \
                                                                        \
    __gmp_i = (ysize);                                                  \
    if (__gmp_i != 0)                                                   \
      {                                                                 \
        if (FUNCTION (wp, xp, yp, __gmp_i))                             \
          {                                                             \
            do                                                          \
              {                                                         \
                if (__gmp_i >= (xsize))                                 \
                  {                                                     \
                    (cout) = 1;                                         \
                    goto __gmp_done;                                    \
                  }                                                     \
                __gmp_x = (xp)[__gmp_i];                                \
              }                                                         \
            while (TEST);                                               \
          }                                                             \
      }                                                                 \
    if ((wp) != (xp))                                                   \
      __GMPN_COPY_REST (wp, xp, xsize, __gmp_i);                        \
    (cout) = 0;                                                         \
  __gmp_done:                                                           \
    ;                                                                   \
  } while (0)

#define __GMPN_ADD(cout, wp, xp, xsize, yp, ysize)              \
  __GMPN_AORS (cout, wp, xp, xsize, yp, ysize, PREFIX(mpn_add_n),       \
               (((wp)[__gmp_i++] = (__gmp_x + 1) & GPU_GMP_NUMB_MASK) == 0))
#define __GMPN_SUB(cout, wp, xp, xsize, yp, ysize)              \
  __GMPN_AORS (cout, wp, xp, xsize, yp, ysize, PREFIX(mpn_sub_n),       \
               (((wp)[__gmp_i++] = (__gmp_x - 1) & GPU_GMP_NUMB_MASK), __gmp_x == 0))

#define __GMPN_AORS_STRIDED(cout, wp, ws, xp, xsize, xs, yp, ysize, ys, FUNCTION, TEST)     \
  do {                                                                  \
    mp_size_t  __gmp_i;                                                 \
    mp_limb_t  __gmp_x;                                                 \
                                                                        \
    __gmp_i = (ysize);                                                  \
    if (__gmp_i != 0)                                                   \
      {                                                                 \
        if (FUNCTION (wp, ws, xp, xs, yp, ys, __gmp_i))                 \
          {                                                             \
            do                                                          \
              {                                                         \
                if (__gmp_i >= (xsize))                                 \
                  {                                                     \
                    (cout) = 1;                                         \
                    goto __gmp_done;                                    \
                  }                                                     \
                __gmp_x = (xp)[__gmp_i * (xs)];                         \
              }                                                         \
            while (TEST);                                               \
          }                                                             \
      }                                                                 \
    if ((wp) != (xp))                                                   \
      __GMPN_COPY_REST_STRIDED (wp, ws, xp, xsize, xs, __gmp_i);        \
    (cout) = 0;                                                         \
  __gmp_done:                                                           \
    ;                                                                   \
  } while (0)

#define __GMPN_ADD_STRIDED(cout, wp, ws, xp, xsize, xs, yp, ysize, ys)  \
  __GMPN_AORS_STRIDED (cout, wp, ws, xp, xsize, xs, yp, ysize, ys, PREFIX(mpn_add_n_strided),       \
                       (((wp)[(__gmp_i++) * (ws)] = (__gmp_x + 1) & GPU_GMP_NUMB_MASK) == 0))
#define __GMPN_SUB_STRIDED(cout, wp, ws, xp, xsize, xs, yp, ysize, ys)  \
  __GMPN_AORS_STRIDED (cout, wp, ws, xp, xsize, xs, yp, ysize, ys, PREFIX(mpn_sub_n_strided),       \
                       (((wp)[(__gmp_i++) * (ws)] = (__gmp_x - 1) & GPU_GMP_NUMB_MASK), __gmp_x == 0))

#define __GMPN_AORS_1(cout, dst, src, n, v, OP, CB)                \
  do {                                                                \
    mp_size_t  __gmp_i;                                                \
    mp_limb_t  __gmp_x, __gmp_r;                                \
                                                                \
    ASSERT ((n) >= 1);                                          \
    ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, n));              \
                                                                \
    __gmp_x = (src)[0];                                                \
    __gmp_r = __gmp_x OP (v);                                   \
    (dst)[0] = __gmp_r;                                                \
    if (CB (__gmp_r, __gmp_x, (v)))                             \
      {                                                                \
        (cout) = 1;                                                \
        for (__gmp_i = 1; __gmp_i < (n);)                       \
          {                                                        \
            __gmp_x = (src)[__gmp_i];                           \
            __gmp_r = __gmp_x OP 1;                             \
            (dst)[__gmp_i] = __gmp_r;                           \
            ++__gmp_i;                                                \
            if (!CB (__gmp_r, __gmp_x, 1))                      \
              {                                                        \
                if ((src) != (dst))                                \
                  __GMPN_COPY_REST (dst, src, n, __gmp_i);      \
                (cout) = 0;                                        \
                break;                                                \
              }                                                        \
          }                                                        \
      }                                                                \
    else                                                        \
      {                                                                \
        if ((src) != (dst))                                        \
          __GMPN_COPY_REST (dst, src, n, 1);                        \
        (cout) = 0;                                                \
      }                                                                \
  } while (0)

#define __GMPN_ADDCB(r,x,y) ((r) < (y))
#define __GMPN_SUBCB(r,x,y) ((x) < (y))

#define __GMPN_ADD_1(cout, dst, src, n, v)             \
  __GMPN_AORS_1(cout, dst, src, n, v, +, __GMPN_ADDCB)
#define __GMPN_SUB_1(cout, dst, src, n, v)             \
  __GMPN_AORS_1(cout, dst, src, n, v, -, __GMPN_SUBCB)

/* Compare {xp,size} and {yp,size}, setting "result" to positive, zero or
   negative.  size==0 is allowed.  On random data usually only one limb will
   need to be examined to get a result, so it's worth having it inline.  */
#define __GMPN_CMP(result, xp, yp, size)                                \
  do {                                                                  \
    mp_size_t  __gmp_i;                                                 \
    mp_limb_t  __gmp_x, __gmp_y;                                        \
                                                                        \
    /* ASSERT ((size) >= 0); */                                         \
                                                                        \
    (result) = 0;                                                       \
    __gmp_i = (size);                                                   \
    while (--__gmp_i >= 0)                                              \
      {                                                                 \
        __gmp_x = (xp)[__gmp_i];                                        \
        __gmp_y = (yp)[__gmp_i];                                        \
        if (__gmp_x != __gmp_y)                                         \
          {                                                             \
            /* Cannot use __gmp_x - __gmp_y, may overflow an "int" */   \
            (result) = (__gmp_x > __gmp_y ? 1 : -1);                    \
            break;                                                      \
          }                                                             \
      }                                                                 \
  } while (0)

#define __GMPN_CMP_STRIDED(result, xp, xs, yp, ys, size)                \
  do {                                                                  \
    mp_size_t  __gmp_i;                                                 \
    mp_limb_t  __gmp_x, __gmp_y;                                        \
                                                                        \
    (result) = 0;                                                       \
    __gmp_i = (size);                                                   \
    while (--__gmp_i >= 0)                                              \
      {                                                                 \
        __gmp_x = (xp)[__gmp_i * (xs)];                                 \
        __gmp_y = (yp)[__gmp_i * (ys)];                                 \
        if (__gmp_x != __gmp_y)                                         \
          {                                                             \
            /* Cannot use __gmp_x - __gmp_y, may overflow an "int" */   \
            (result) = (__gmp_x > __gmp_y ? 1 : -1);                    \
            break;                                                      \
          }                                                             \
      }                                                                 \
  } while (0)

__all__ mp_limb_t
PREFIX(mpn_add) (mp_ptr wp, mp_srcptr xp, mp_size_t xsize, mp_srcptr yp, mp_size_t ysize)
{
  mp_limb_t  c;
  __GMPN_ADD (c, wp, xp, xsize, yp, ysize);
  return c;
}

__all__ mp_limb_t
PREFIX(mpn_add_strided) (mp_ptr wp, mp_size_t ws,
		mp_srcptr xp, mp_size_t xsize, mp_size_t xs,
		mp_srcptr yp, mp_size_t ysize, mp_size_t ys)
{
  mp_limb_t  c;
  __GMPN_ADD_STRIDED (c, wp, ws, xp, xsize, xs, yp, ysize, ys);
  return c;
}

// The below version is an attempt to reduce divergent execution paths.
// The conditional branch on the return value of FUNCTION had divergent
// behaviour many times according to the profiler, so it was eliminated;
// however, performance did not increase, so the old version is kept.
/*__all__ mp_limb_t
PREFIX(mpn_add_strided) (mp_ptr wp, mp_size_t ws,
		mp_srcptr xp, mp_size_t xsize, mp_size_t xs,
		mp_srcptr yp, mp_size_t ysize, mp_size_t ys)
{
  mp_size_t  i;
  mp_limb_t  x;

  i = ysize;
  if (i != 0)
    {
      mp_limb_t c = PREFIX(mpn_add_n_strided) (wp, ws, xp, xs, yp, ys, i);
      do
        {
          if (i >= xsize)
            return c;
          x = xp[i * xs] + c;
          wp[i * ws] = x;
          i++;
        }
      while (x == 0);
    }
  if (wp != xp)
    __GMPN_COPY_REST_STRIDED (wp, ws, xp, xsize, xs, i);
  return 0;
}*/

__all__ mp_limb_t
PREFIX(mpn_add_1) (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t n) __GMP_NOTHROW
{
  mp_limb_t  c;
  __GMPN_ADD_1 (c, dst, src, size, n);
  return c;
}

__all__ int
PREFIX(mpn_cmp) (mp_srcptr xp, mp_srcptr yp, mp_size_t size) __GMP_NOTHROW
{
  int result;
  __GMPN_CMP (result, xp, yp, size);
  return result;
}

__all__ int
PREFIX(mpn_cmp_strided) (mp_srcptr xp, mp_size_t xs, mp_srcptr yp, mp_size_t ys, mp_size_t size) __GMP_NOTHROW
{
  int result;
  __GMPN_CMP_STRIDED (result, xp, xs, yp, ys, size);
  return result;
}

__all__ int
PREFIX(mpn_zero_p) (mp_srcptr p, mp_size_t n) __GMP_NOTHROW
{
  // Below comment was already present in original gmp.h
  /* if (__GMP_LIKELY (n > 0)) */
    do {
      if (p[--n] != 0)
        return 0;
    } while (n != 0);
  return 1;
}

__all__ int
PREFIX(mpn_zero_p_strided) (mp_srcptr p, mp_size_t n, mp_size_t s) __GMP_NOTHROW
{
  // Below comment was already present in original gmp.h
  /* if (__GMP_LIKELY (n > 0)) */
    do {
      if (p[--n * s] != 0)
        return 0;
    } while (n != 0);
  return 1;
}

__all__ mp_limb_t
PREFIX(mpn_sub) (mp_ptr wp, mp_srcptr xp, mp_size_t xsize, mp_srcptr yp, mp_size_t ysize)
{
  mp_limb_t  c;
  __GMPN_SUB (c, wp, xp, xsize, yp, ysize);
  return c;
}

__all__ mp_limb_t
PREFIX(mpn_sub_strided) (mp_ptr wp, mp_size_t ws,
      mp_srcptr xp, mp_size_t xsize, mp_size_t xs,
      mp_srcptr yp, mp_size_t ysize, mp_size_t ys)
{
  mp_limb_t  c;
  __GMPN_SUB_STRIDED (c, wp, ws, xp, xsize, xs, yp, ysize, ys);
  return c;
}

__all__ mp_limb_t
PREFIX(mpn_sub_1) (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t n) __GMP_NOTHROW
{
  mp_limb_t  c;
  __GMPN_SUB_1 (c, dst, src, size, n);
  return c;
}

mp_limb_t
PREFIX(mpn_neg) (mp_ptr rp, mp_srcptr up, mp_size_t n)
{
  while (*up == 0) /* Low zero limbs are unchanged by negation. */
    {
      *rp = 0;
      if (!--n) /* All zero */
        return 0;
      ++up; ++rp;
    }

  *rp = (- *up) & GPU_GMP_NUMB_MASK;

  if (--n) /* Higher limbs get complemented. */
    PREFIX(mpn_com) (++rp, ++up, n);

  return 1;
}

// vim: set sw=2 ts=8 noet:
