/* mpn_mul_n -- multiply natural numbers.

Copyright 1991, 1993, 1994, 1996-2003, 2005, 2008, 2009 Free Software
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

#include "gmp-es.h"
#include "gmp-es-internal-convert.h"

__all__ void
PREFIX(mpn_mul_n) (mp_ptr p, mp_srcptr a, mp_srcptr b, mp_size_t n)
{
  ASSERT (n >= 1);
  ASSERT (! MPN_OVERLAP_P (p, 2 * n, a, n));
  ASSERT (! MPN_OVERLAP_P (p, 2 * n, b, n));

  if (BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))
    {
      PREFIX(mpn_mul_basecase) (p, a, n, b, n);
    }
  else if (BELOW_THRESHOLD (n, MUL_TOOM33_THRESHOLD))
    {
      /* Allocate workspace of fixed size on stack: fast! */
      mp_limb_t ws[mpn_toom22_mul_itch (MUL_TOOM33_THRESHOLD_LIMIT-1,
                                        MUL_TOOM33_THRESHOLD_LIMIT-1)];
      ASSERT (MUL_TOOM33_THRESHOLD <= MUL_TOOM33_THRESHOLD_LIMIT);
      PREFIX(mpn_toom22_mul) (p, a, n, b, n, ws);
    }
  else if (BELOW_THRESHOLD (n, MUL_TOOM44_THRESHOLD))
    {
      UNIMPLEMENTED;
#if 0
      mp_ptr ws;
      TMP_SDECL;
      TMP_SMARK;
      ws = TMP_SALLOC_LIMBS (mpn_toom33_mul_itch (n, n));
      mpn_toom33_mul (p, a, n, b, n, ws);
      TMP_SFREE;
#endif
    }
  else if (BELOW_THRESHOLD (n, MUL_TOOM6H_THRESHOLD))
    {
      UNIMPLEMENTED;
#if 0
      mp_ptr ws;
      TMP_SDECL;
      TMP_SMARK;
      ws = TMP_SALLOC_LIMBS (mpn_toom44_mul_itch (n, n));
      mpn_toom44_mul (p, a, n, b, n, ws);
      TMP_SFREE;
#endif
    }
  else if (BELOW_THRESHOLD (n, MUL_TOOM8H_THRESHOLD))
    {
      UNIMPLEMENTED;
#if 0
      mp_ptr ws;
      TMP_SDECL;
      TMP_SMARK;
      ws = TMP_SALLOC_LIMBS (mpn_toom6_mul_n_itch (n));
      mpn_toom6h_mul (p, a, n, b, n, ws);
      TMP_SFREE;
#endif
    }
  else if (BELOW_THRESHOLD (n, MUL_FFT_THRESHOLD))
    {
      UNIMPLEMENTED;
#if 0
      mp_ptr ws;
      TMP_DECL;
      TMP_MARK;
      ws = TMP_ALLOC_LIMBS (mpn_toom8_mul_n_itch (n));
      mpn_toom8h_mul (p, a, n, b, n, ws);
      TMP_FREE;
#endif
    }
  else
    {
      UNIMPLEMENTED;
#if 0
      /* The current FFT code allocates its own space.  That should probably
         change.  */
      mpn_fft_mul (p, a, n, b, n);
#endif
    }
}

__all__ void
PREFIX(mpn_mul_n_strided) (mp_ptr p, mp_size_t ps, mp_srcptr a, mp_size_t as, mp_srcptr b, mp_size_t bs, mp_size_t n)
{
  if (BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))
    {
      PREFIX(mpn_mul_basecase_strided) (p, ps, a, n, as, b, n, bs);
    }
  else if (BELOW_THRESHOLD (n, MUL_TOOM33_THRESHOLD))
    {
      /* Allocate workspace of fixed size on stack: fast! */
      mp_limb_t scratch[mpn_toom22_mul_itch (MUL_TOOM33_THRESHOLD_LIMIT-1,
                                             MUL_TOOM33_THRESHOLD_LIMIT-1)];
      ASSERT (MUL_TOOM33_THRESHOLD <= MUL_TOOM33_THRESHOLD_LIMIT);
      PREFIX(mpn_toom22_mul_strided) (p, ps, a, n, as, b, n, bs, scratch, 1);
    }
  else if (BELOW_THRESHOLD (n, MUL_TOOM44_THRESHOLD))
    {
      UNIMPLEMENTED;
#if 0
      mp_ptr ws;
      TMP_SDECL;
      TMP_SMARK;
      ws = TMP_SALLOC_LIMBS (mpn_toom33_mul_itch (n, n));
      mpn_toom33_mul (p, a, n, b, n, ws);
      TMP_SFREE;
#endif
    }
  else if (BELOW_THRESHOLD (n, MUL_TOOM6H_THRESHOLD))
    {
      UNIMPLEMENTED;
#if 0
      mp_ptr ws;
      TMP_SDECL;
      TMP_SMARK;
      ws = TMP_SALLOC_LIMBS (mpn_toom44_mul_itch (n, n));
      mpn_toom44_mul (p, a, n, b, n, ws);
      TMP_SFREE;
#endif
    }
  else if (BELOW_THRESHOLD (n, MUL_TOOM8H_THRESHOLD))
    {
      UNIMPLEMENTED;
#if 0
      mp_ptr ws;
      TMP_SDECL;
      TMP_SMARK;
      ws = TMP_SALLOC_LIMBS (mpn_toom6_mul_n_itch (n));
      mpn_toom6h_mul (p, a, n, b, n, ws);
      TMP_SFREE;
#endif
    }
  else if (BELOW_THRESHOLD (n, MUL_FFT_THRESHOLD))
    {
      UNIMPLEMENTED;
#if 0
      mp_ptr ws;
      TMP_DECL;
      TMP_MARK;
      ws = TMP_ALLOC_LIMBS (mpn_toom8_mul_n_itch (n));
      mpn_toom8h_mul (p, a, n, b, n, ws);
      TMP_FREE;
#endif
    }
  else
    {
      UNIMPLEMENTED;
#if 0
      /* The current FFT code allocates its own space.  That should probably
         change.  */
      mpn_fft_mul (p, a, n, b, n);
#endif
    }
}

// vim: set sw=2 ts=8 noet:
