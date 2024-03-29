/* mpn_addmul_1 -- multiply the N long limb vector pointed to by UP by VL,
   add the N least significant limbs of the product to the limb vector
   pointed to by RP.  Return the most significant limb of the product,
   adjusted for carry-out from the addition.

Copyright 1992-1994, 1996, 2000, 2002, 2004 Free Software Foundation, Inc.

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


__all__ mp_limb_t
PREFIX(mpn_addmul_1) (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t vl)
{
  mp_limb_t ul, cl, hpl, lpl, rl;

  ASSERT (n >= 1);
  ASSERT (MPN_SAME_OR_SEPARATE_P (rp, up, n));

  cl = 0;
  do
    {
      ul = *up++;
      umul_ppmm (hpl, lpl, ul, vl);

      lpl += cl;
      cl = (lpl < cl) + hpl;

      rl = *rp;
      lpl = rl + lpl;
      cl += lpl < rl;
      *rp++ = lpl;
    }
  while (--n != 0);

  return cl;
}

__all__ mp_limb_t
PREFIX(mpn_addmul_1_strided) (mp_ptr rp, mp_size_t rs, mp_srcptr up, mp_size_t n, mp_size_t us, mp_limb_t vl)
{
  mp_limb_t ul, cl, hpl, lpl, rl;

  cl = 0;
  do
    {
      ul = *up; up += us;
      umul_ppmm (hpl, lpl, ul, vl);

      lpl += cl;
      cl = (lpl < cl) + hpl;

      rl = *rp;
      lpl = rl + lpl;
      cl += lpl < rl;
      *rp = lpl; rp += rs;
    }
  while (--n != 0);

  return cl;
}

// vim: set sw=2 ts=8 noet:
