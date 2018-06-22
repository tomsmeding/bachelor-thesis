/* mpz_add, mpz_sub -- add or subtract integers.

Copyright 1991, 1993, 1994, 1996, 2000, 2001, 2011, 2012 Free Software
Foundation, Inc.

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

#include "gmp-es-mpz.h"
#include "gmp-es-internal-convert.h"


#ifdef DEFINE_SUB
#define FUNCTION     PREFIX(mpz_sub)
#define VARIATION    -
#else
#define FUNCTION     PREFIX(mpz_add)
#define VARIATION
#endif

#define CAT_(a, b) a ## b
#define CAT(a, b) CAT_(a, b)


__all__ void
FUNCTION (mpz_ptr w, mpz_srcptr u, mpz_srcptr v)
{
  mp_srcptr up, vp;
  mp_ptr wp;
  mp_size_t usize, vsize, wsize;
  mp_size_t abs_usize;
  mp_size_t abs_vsize;

  usize = SIZ(u);
  vsize = VARIATION SIZ(v);
  abs_usize = ABS (usize);
  abs_vsize = ABS (vsize);

  if (abs_usize < abs_vsize)
    {
      /* Swap U and V. */
      MPZ_SRCPTR_SWAP (u, v);
      MP_SIZE_T_SWAP (usize, vsize);
      MP_SIZE_T_SWAP (abs_usize, abs_vsize);
    }

  /* True: ABS_USIZE >= ABS_VSIZE.  */

  /* If not space for w (and possible carry), increase space.  */
  wsize = abs_usize + 1;
  wp = MPZ_REALLOC (w, wsize);

  /* These must be after realloc (u or v may be the same as w).  */
  up = PTR(u);
  vp = PTR(v);

  if ((usize ^ vsize) < 0)
    {
      /* U and V have different sign.  Need to compare them to determine
	 which operand to subtract from which.  */

      /* This test is right since ABS_USIZE >= ABS_VSIZE.  */
      if (abs_usize != abs_vsize)
	{
	  PREFIX(mpn_sub) (wp, up, abs_usize, vp, abs_vsize);
	  wsize = abs_usize;
	  MPN_NORMALIZE (wp, wsize);
	  if (usize < 0)
	    wsize = -wsize;
	}
      else if (PREFIX(mpn_cmp) (up, vp, abs_usize) < 0)
	{
	  PREFIX(mpn_sub_n) (wp, vp, up, abs_usize);
	  wsize = abs_usize;
	  MPN_NORMALIZE (wp, wsize);
	  if (usize >= 0)
	    wsize = -wsize;
	}
      else
	{
	  PREFIX(mpn_sub_n) (wp, up, vp, abs_usize);
	  wsize = abs_usize;
	  MPN_NORMALIZE (wp, wsize);
	  if (usize < 0)
	    wsize = -wsize;
	}
    }
  else
    {
      /* U and V have same sign.  Add them.  */
      mp_limb_t cy_limb = PREFIX(mpn_add) (wp, up, abs_usize, vp, abs_vsize);
      wp[abs_usize] = cy_limb;
      wsize = abs_usize + cy_limb;
      if (usize < 0)
	wsize = -wsize;
    }

  SIZ(w) = wsize;
}

__all__ void
CAT(FUNCTION, _strided) (mpz_ptr w, mp_size_t ws, mpz_srcptr u, mp_size_t us, mpz_srcptr v, mp_size_t vs)
{
  mp_srcptr up, vp;
  mp_ptr wp;
  mp_size_t usize, vsize, wsize;
  mp_size_t abs_usize;
  mp_size_t abs_vsize;

  usize = STRIDED_SIZE(u, us);
  vsize = VARIATION STRIDED_SIZE(v, vs);
  abs_usize = ABS (usize);
  abs_vsize = ABS (vsize);

  if (abs_usize < abs_vsize)
    {
      /* Swap U and V. */
      MPZ_SRCPTR_SWAP (u, v);
      MP_SIZE_T_SWAP (usize, vsize);
      MP_SIZE_T_SWAP (abs_usize, abs_vsize);
    }

  /* True: ABS_USIZE >= ABS_VSIZE.  */

  /* If not space for w (and possible carry), increase space.  */
  wsize = abs_usize + 1;
  wp = MPZ_REALLOC_STRIDED (w, wsize);

  /* These must be after realloc (u or v may be the same as w).  */
  up = STRIDED_PTR(u);
  vp = STRIDED_PTR(v);

  if ((usize ^ vsize) < 0)
    {
      /* U and V have different sign.  Need to compare them to determine
	 which operand to subtract from which.  */

      /* This test is right since ABS_USIZE >= ABS_VSIZE.  */
      if (abs_usize != abs_vsize)
	{
	  PREFIX(mpn_sub_strided) (wp, ws, up, abs_usize, us, vp, abs_vsize, vs);
	  wsize = abs_usize;
	  MPN_NORMALIZE_STRIDED (wp, wsize, ws);
	  if (usize < 0)
	    wsize = -wsize;
	}
      else if (PREFIX(mpn_cmp_strided) (up, us, vp, vs, abs_usize) < 0)
	{
	  PREFIX(mpn_sub_n_strided) (wp, ws, vp, vs, up, us, abs_usize);
	  wsize = abs_usize;
	  MPN_NORMALIZE_STRIDED (wp, wsize, ws);
	  if (usize >= 0)
	    wsize = -wsize;
	}
      else
	{
	  PREFIX(mpn_sub_n_strided) (wp, ws, up, us, vp, vs, abs_usize);
	  wsize = abs_usize;
	  MPN_NORMALIZE_STRIDED (wp, wsize, ws);
	  if (usize < 0)
	    wsize = -wsize;
	}
    }
  else
    {
      /* U and V have same sign.  Add them.  */
      mp_limb_t cy_limb = PREFIX(mpn_add_strided) (wp, ws, up, abs_usize, us, vp, abs_vsize, vs);
      wp[abs_usize * ws] = cy_limb;
      wsize = abs_usize + cy_limb;
      if (usize < 0)
	wsize = -wsize;
    }

  STRIDED_SIZE(w, ws) = wsize;
}

// vim: set sw=2 ts=8 noet:
