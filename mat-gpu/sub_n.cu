/* mpn_sub_n -- Subtract equal length limb vectors.

Copyright 1992-1994, 1996, 2000, 2002, 2009 Free Software Foundation, Inc.

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
PREFIX(mpn_sub_n) (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
{
  mp_limb_t ul, vl, sl, rl, cy, cy1, cy2;

  ASSERT (n >= 1);
  ASSERT (MPN_SAME_OR_INCR_P (rp, up, n));
  ASSERT (MPN_SAME_OR_INCR_P (rp, vp, n));

  cy = 0;
  do
    {
      ul = *up++;
      vl = *vp++;
      sl = ul - vl;
      cy1 = sl > ul;
      rl = sl - cy;
      cy2 = rl > sl;
      cy = cy1 | cy2;
      *rp++ = rl;
    }
  while (--n != 0);

  return cy;
}

__all__ mp_limb_t
PREFIX(mpn_sub_n_strided) (mp_ptr rp, mp_size_t rs,
    mp_srcptr up, mp_size_t us,
    mp_srcptr vp, mp_size_t vs, mp_size_t n)
{
  mp_limb_t ul, vl, sl, rl, cy, cy1, cy2;

  cy = 0;
  do
    {
      ul = *up; up += us;
      vl = *vp; vp += vs;
      sl = ul - vl;
      cy1 = sl > ul;
      rl = sl - cy;
      cy2 = rl > sl;
      cy = cy1 | cy2;
      *rp = rl; rp += rs;
    }
  while (--n != 0);

  return cy;
}

// vim: set sw=2 ts=8 noet:
