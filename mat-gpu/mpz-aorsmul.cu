/* mpz_addmul, mpz_submul -- add or subtract multiple.

Copyright 2001, 2004, 2005, 2012 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include <assert.h>
#include "gmp-es-mpz.h"
#include "gmp-es-internal-convert.h"


#ifdef DEFINE_BASE
#define SUFFIX_(name_) name_ ## _base
#define SUFFIX_STRIDED_(name_) name_ ## _base_strided
#define OPERATION(wp_, wn_, up_, un_, vp_, vn_) (PREFIX(mpn_mul_basecase) ((wp_), (up_), (un_), (vp_), (vn_)), (wp_)[(wn_) - 1])
#define OPERATION_STRIDED(wp_, wn_, ws_, up_, un_, us_, vp_, vn_, vs_) (PREFIX(mpn_mul_basecase_strided) ((wp_), (ws_), (up_), (un_), (us_), (vp_), (vn_), (vs_)), (wp_)[((wn_) - 1) * (ws_)])
#else
#define SUFFIX_(name_) name_
#define SUFFIX_STRIDED_(name_) name_ ## _strided
#define OPERATION(wp_, wn_, up_, un_, vp_, vn_) (PREFIX(mpn_mul) ((wp_), (up_), (un_), (vp_), (vn_)))
#define OPERATION_STRIDED(wp_, wn_, ws_, up_, un_, us_, vp_, vn_, vs_) (PREFIX(mpn_mul_strided) ((wp_), (ws_), (up_), (un_), (us_), (vp_), (vn_), (vs_)))
#endif

#define SUFFIX(name_) SUFFIX_(name_)
#define SUFFIX_STRIDED(name_) SUFFIX_STRIDED_(name_)


/* expecting x and y both with non-zero high limbs */
#define mpn_cmp_twosizes_lt(xp,xsize, yp,ysize)                 \
  ((xsize) < (ysize)                                            \
   || ((xsize) == (ysize) && PREFIX(mpn_cmp) (xp, yp, xsize) < 0))

#define mpn_cmp_twosizes_lt_strided(xp,xsize,xs, yp,ysize,ys)                 \
  ((xsize) < (ysize)                                            \
   || ((xsize) == (ysize) && PREFIX(mpn_cmp_strided) (xp, xs, yp, ys, xsize) < 0))


/* sub>=0 means an addmul w += x*y, sub<0 means a submul w -= x*y.

   The signs of w, x and y are fully accounted for by each flipping "sub".

   The sign of w is retained for the result, unless the absolute value
   submul underflows, in which case it flips.  */

__all__ static inline void
SUFFIX(aorsmul) (mpz_ptr w, mpz_srcptr x, mpz_srcptr y, mpz_ptr scratch, mp_size_t sub)
{
  mp_size_t  xsize, ysize, tsize, wsize, wsize_signed;
  mp_ptr     wp, tp;
  mp_limb_t  c, high;

  /* w unaffected if x==0 or y==0 */
  xsize = SIZ(x);
  ysize = SIZ(y);
  if (xsize == 0 || ysize == 0)
    return;

  /* make x the bigger of the two */
  if (ABS(ysize) > ABS(xsize))
    {
      MPZ_SRCPTR_SWAP (x, y);
      MP_SIZE_T_SWAP (xsize, ysize);
    }

  sub ^= ysize;
  ysize = ABS(ysize);

  // aorsmul_1 elided

  sub ^= xsize;
  xsize = ABS(xsize);

  wsize_signed = SIZ(w);
  sub ^= wsize_signed;
  wsize = ABS(wsize_signed);

  tsize = xsize + ysize;
  wp = MPZ_REALLOC (w, MAX (wsize, tsize) + 1);

  if (wsize_signed == 0)
    {
      /* Nothing to add to, just set w=x*y.  No w==x or w==y overlap here,
	 since we know x,y!=0 but w==0.  */
      high = OPERATION (wp, tsize, PTR(x),xsize, PTR(y),ysize);
      tsize -= (high == 0);
      SIZ(w) = (sub >= 0 ? tsize : -tsize);
      return;
    }

  tp = PTR (scratch);
  assert (ALLOC (scratch) >= tsize);

  high = OPERATION (tp, tsize, PTR(x),xsize, PTR(y),ysize);
  tsize -= (high == 0);
  ASSERT (tp[tsize-1] != 0);
  if (sub >= 0)
    {
      mp_srcptr up    = wp;
      mp_size_t usize = wsize;

      if (usize < tsize)
	{
	  up	= tp;
	  usize = tsize;
	  tp	= wp;
	  tsize = wsize;

	  wsize = usize;
	}

      c = PREFIX(mpn_add) (wp, up,usize, tp,tsize);
      wp[wsize] = c;
      wsize += (c != 0);
    }
  else
    {
      mp_srcptr up    = wp;
      mp_size_t usize = wsize;

      if (mpn_cmp_twosizes_lt (up,usize, tp,tsize))
	{
	  up	= tp;
	  usize = tsize;
	  tp	= wp;
	  tsize = wsize;

	  wsize = usize;
	  wsize_signed = -wsize_signed;
	}

      ASSERT_NOCARRY (PREFIX(mpn_sub) (wp, up,usize, tp,tsize));
      wsize = usize;
      MPN_NORMALIZE (wp, wsize);
    }

  SIZ(w) = (wsize_signed >= 0 ? wsize : -wsize);
}

__all__ static inline void
SUFFIX_STRIDED(aorsmul) (mpz_ptr w, mp_size_t ws, mpz_srcptr x, mp_size_t xs, mpz_srcptr y, mp_size_t ys, mpz_ptr scratch, mp_size_t ss, mp_size_t sub)
{
  mp_size_t  xsize, ysize, tsize, wsize, wsize_signed;
  mp_ptr     wp, tp;
  mp_limb_t  c, high;

  /* w unaffected if x==0 or y==0 */
  xsize = STRIDED_SIZE(x, xs);
  ysize = STRIDED_SIZE(y, ys);
  if (xsize == 0 || ysize == 0)
    return;

  /* make x the bigger of the two */
  if (ABS(ysize) > ABS(xsize))
    {
      MPZ_SRCPTR_SWAP (x, y);
      MP_SIZE_T_SWAP (xsize, ysize);
    }

  sub ^= ysize;
  ysize = ABS(ysize);

  // aorsmul_1 elided

  sub ^= xsize;
  xsize = ABS(xsize);

  wsize_signed = STRIDED_SIZE(w, ws);
  sub ^= wsize_signed;
  wsize = ABS(wsize_signed);

  tsize = xsize + ysize;
  wp = MPZ_REALLOC_STRIDED (w, MAX (wsize, tsize) + 1);

  if (wsize_signed == 0)
    {
      /* Nothing to add to, just set w=x*y.  No w==x or w==y overlap here,
	 since we know x,y!=0 but w==0.  */
      high = OPERATION_STRIDED (wp, tsize, ws, STRIDED_PTR(x),xsize,xs, STRIDED_PTR(y),ysize,ys);
      tsize -= (high == 0);
      STRIDED_SIZE(w, ws) = (sub >= 0 ? tsize : -tsize);
      return;
    }

  tp = STRIDED_PTR (scratch);
  assert (STRIDED_ALLOC (scratch) >= tsize);

  high = OPERATION_STRIDED (tp, tsize, ss, STRIDED_PTR(x),xsize,xs, STRIDED_PTR(y),ysize,ys);
  tsize -= (high == 0);
  ASSERT (tp[(tsize-1) * ss] != 0);
  if (sub >= 0)
    {
      mp_srcptr up    = wp;
      mp_size_t usize = wsize;
      mp_size_t us = ws;

      mp_size_t ts = ss;

      if (usize < tsize)
	{
	  up	= tp;
	  usize = tsize;
	  us    = ts;
	  tp	= wp;
	  tsize = wsize;
	  ts    = ws;

	  wsize = usize;
	}

      c = PREFIX(mpn_add_strided) (wp,ws, up,usize,us, tp,tsize,ts);
      wp[wsize * ws] = c;
      wsize += (c != 0);
    }
  else
    {
      mp_srcptr up    = wp;
      mp_size_t usize = wsize;
      mp_size_t us = ws;

      mp_size_t ts = ss;

      if (mpn_cmp_twosizes_lt_strided (up,usize,us, tp,tsize,ts))
	{
	  up	= tp;
	  usize = tsize;
	  us    = ts;
	  tp	= wp;
	  tsize = wsize;
	  ts    = ws;

	  wsize = usize;
	  wsize_signed = -wsize_signed;
	}

      ASSERT_NOCARRY (PREFIX(mpn_sub_strided) (wp,ws, up,usize,us, tp,tsize,ts));
      wsize = usize;
      MPN_NORMALIZE_STRIDED (wp, wsize, ws);
    }

  STRIDED_SIZE(w, ws) = (wsize_signed >= 0 ? wsize : -wsize);
}


__all__ void
SUFFIX(PREFIX(mpz_addmul)) (mpz_ptr w, mpz_srcptr u, mpz_srcptr v, mpz_ptr scratch)
{
  SUFFIX(aorsmul) (w, u, v, scratch, (mp_size_t) 0);
}

__all__ void
SUFFIX_STRIDED(PREFIX(mpz_addmul)) (mpz_ptr w, mp_size_t ws, mpz_srcptr u, mp_size_t us, mpz_srcptr v, mp_size_t vs, mpz_ptr scratch, mp_size_t ss)
{
  SUFFIX_STRIDED(aorsmul) (w, ws, u, us, v, vs, scratch, ss, (mp_size_t) 0);
}

__all__ void
SUFFIX(PREFIX(mpz_submul)) (mpz_ptr w, mpz_srcptr u, mpz_srcptr v, mpz_ptr scratch)
{
  SUFFIX(aorsmul) (w, u, v, scratch, (mp_size_t) -1);
}

__all__ void
SUFFIX_STRIDED(PREFIX(mpz_submul)) (mpz_ptr w, mp_size_t ws, mpz_srcptr u, mp_size_t us, mpz_srcptr v, mp_size_t vs, mpz_ptr scratch, mp_size_t ss)
{
  SUFFIX_STRIDED(aorsmul) (w, ws, u, us, v, vs, scratch, ss, (mp_size_t) -1);
}

// vim: set sw=2 ts=8 noet:
