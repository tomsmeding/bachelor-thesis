/* mpz_mul -- Multiply two integers.

Copyright 1991, 1993, 1994, 1996, 2000, 2001, 2005, 2009, 2011, 2012 Free
Software Foundation, Inc.

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
#define FUNCTION PREFIX(mpz_mul_base)
#define FUNCTION_STRIDED PREFIX(mpz_mul_base_strided)
#define OPERATION(wp_, wn_, up_, un_, vp_, vn_) (PREFIX(mpn_mul_basecase) ((wp_), (up_), (un_), (vp_), (vn_)), (wp_)[(wn_) - 1])
#define OPERATION_STRIDED(wp_, wn_, ws_, up_, un_, us_, vp_, vn_, vs_) (PREFIX(mpn_mul_basecase_strided) ((wp_), (ws_), (up_), (un_), (us_), (vp_), (vn_), (vs_)), (wp_)[((wn_) - 1) * (ws_)])
#else
#define FUNCTION PREFIX(mpz_mul)
#define FUNCTION_STRIDED PREFIX(mpz_mul_strided)
#define OPERATION(wp_, wn_, up_, un_, vp_, vn_) (PREFIX(mpn_mul) ((wp_), (up_), (un_), (vp_), (vn_)))
#define OPERATION_STRIDED(wp_, wn_, ws_, up_, un_, us_, vp_, vn_, vs_) (PREFIX(mpn_mul_strided) ((wp_), (ws_), (up_), (un_), (us_), (vp_), (vn_), (vs_)))
#endif


__all__ void
FUNCTION (mpz_ptr w, mpz_srcptr u, mpz_srcptr v)
{
  mp_size_t usize;
  mp_size_t vsize;
  mp_size_t wsize;
  mp_size_t sign_product;
  mp_srcptr up, vp;
  mp_ptr wp;
  mp_limb_t cy_limb;

  usize = SIZ (u);
  vsize = SIZ (v);
  sign_product = usize ^ vsize;
  usize = ABS (usize);
  vsize = ABS (vsize);

  if (usize < vsize)
    {
      MPZ_SRCPTR_SWAP (u, v);
      MP_SIZE_T_SWAP (usize, vsize);
    }

  if (vsize == 0)
    {
      SIZ (w) = 0;
      return;
    }

  if (vsize == 1)
    {
      wp = MPZ_REALLOC (w, usize+1);
      cy_limb = PREFIX(mpn_mul_1) (wp, PTR (u), usize, PTR (v)[0]);
      wp[usize] = cy_limb;
      usize += (cy_limb != 0);
      SIZ (w) = (sign_product >= 0 ? usize : -usize);
      return;
    }

  up = PTR (u);
  vp = PTR (v);
  wp = PTR (w);

  /* Ensure W has space enough to store the result.  */
  wsize = usize + vsize;
  assert (ALLOC (w) >= wsize);

  /* Make U and V not overlap with W.  */
  assert (wp != up);
  assert (wp != vp);  // handling code elided

  assert (up != vp);  // handling code called mpn_sqr, elided
  cy_limb = OPERATION (wp, wsize, up, usize, vp, vsize);

  wsize -= cy_limb == 0;

  SIZ (w) = sign_product < 0 ? -wsize : wsize;
}

__all__ void
FUNCTION_STRIDED (PREFIX(mpz_ptr) w, PREFIX(mp_size_t) ws, PREFIX(mpz_srcptr) u, PREFIX(mp_size_t) us, PREFIX(mpz_srcptr) v, PREFIX(mp_size_t) vs)
{
  mp_size_t usize;
  mp_size_t vsize;
  mp_size_t wsize;
  mp_size_t sign_product;
  mp_srcptr up, vp;
  mp_ptr wp;
  mp_limb_t cy_limb;

  usize = STRIDED_SIZE (u, us);
  vsize = STRIDED_SIZE (v, vs);
  sign_product = usize ^ vsize;
  usize = ABS (usize);
  vsize = ABS (vsize);

  if (usize < vsize)
    {
      MPZ_SRCPTR_SWAP (u, v);
      MP_SIZE_T_SWAP (usize, vsize);
    }

  if (vsize == 0)
    {
      STRIDED_SIZE (w, ws) = 0;
      return;
    }

  if (vsize == 1)
    {
      wp = MPZ_REALLOC_STRIDED (w, usize+1);
      cy_limb = PREFIX(mpn_mul_1_strided) (wp, ws, STRIDED_PTR (u), usize, us, STRIDED_PTR (v)[0]);
      wp[usize * ws] = cy_limb;
      usize += (cy_limb != 0);
      STRIDED_SIZE (w, ws) = (sign_product >= 0 ? usize : -usize);
      return;
    }

  up = STRIDED_PTR (u);
  vp = STRIDED_PTR (v);
  wp = STRIDED_PTR (w);

  /* Ensure W has space enough to store the result.  */
  wsize = usize + vsize;
  assert (STRIDED_ALLOC (w) >= wsize);

  /* Make U and V not overlap with W.  */
  assert (wp != up);
  assert (wp != vp);  // handling code elided

  assert (up != vp);  // handling code called mpn_sqr, elided
  cy_limb = OPERATION_STRIDED (wp, wsize, ws, up, usize, us, vp, vsize, vs);

  wsize -= cy_limb == 0;

  STRIDED_SIZE (w, ws) = sign_product < 0 ? -wsize : wsize;
}

// vim: set sw=2 ts=8 noet:
