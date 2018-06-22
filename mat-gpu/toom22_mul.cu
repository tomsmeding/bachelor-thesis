/* mpn_toom22_mul -- Multiply {ap,an} and {bp,bn} where an >= bn.  Or more
   accurately, bn <= an < 2bn.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2006-2010, 2012, 2014 Free Software Foundation, Inc.

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

/* Evaluate in: -1, 0, +inf

  <-s--><--n-->
   ____ ______
  |_a1_|___a0_|
   |b1_|___b0_|
   <-t-><--n-->

  v0  =  a0     * b0       #   A(0)*B(0)
  vm1 = (a0- a1)*(b0- b1)  #  A(-1)*B(-1)
  vinf=      a1 *     b1   # A(inf)*B(inf)
*/

#if TUNE_PROGRAM_BUILD || WANT_FAT_BINARY
#define MAYBE_mul_toom22   1
#else
#define MAYBE_mul_toom22                                                \
  (MUL_TOOM33_THRESHOLD >= 2 * MUL_TOOM22_THRESHOLD)
#endif

#define TOOM22_MUL_N_REC(p, a, b, n, ws)                                \
  do {                                                                        \
    if (! MAYBE_mul_toom22                                                \
        || BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))                        \
      PREFIX(mpn_mul_basecase) (p, a, n, b, n);                                        \
    else                                                                \
      PREFIX(mpn_toom22_mul) (p, a, n, b, n, ws);                                \
  } while (0)

#define TOOM22_MUL_N_REC_STRIDED(p, ps, a, as, b, bs, n, ws, wss)                                \
  do {                                                                        \
    if (! MAYBE_mul_toom22                                                \
        || BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))                        \
      PREFIX(mpn_mul_basecase_strided) (p, ps, a, n, as, b, n, bs);                                        \
    else                                                                \
      PREFIX(mpn_toom22_mul_strided) (p, ps, a, n, as, b, n, bs, ws, wss);                                \
  } while (0)

/* Normally, this calls mul_basecase or toom22_mul.  But when when the fraction
   MUL_TOOM33_THRESHOLD / MUL_TOOM22_THRESHOLD is large, an initially small
   relative unbalance will become a larger and larger relative unbalance with
   each recursion (the difference s-t will be invariant over recursive calls).
   Therefore, we need to call toom32_mul.  FIXME: Suppress depending on
   MUL_TOOM33_THRESHOLD / MUL_TOOM22_THRESHOLD and on MUL_TOOM22_THRESHOLD.  */
#define TOOM22_MUL_REC(p, a, an, b, bn, ws)                                \
  do {                                                                        \
    if (! MAYBE_mul_toom22                                                \
        || BELOW_THRESHOLD (bn, MUL_TOOM22_THRESHOLD))                        \
      PREFIX(mpn_mul_basecase) (p, a, an, b, bn);                                \
    else if (4 * an < 5 * bn)                                                \
      PREFIX(mpn_toom22_mul) (p, a, an, b, bn, ws);                                \
    else                                                                \
      UNIMPLEMENTED; /* mpn_toom32_mul (p, a, an, b, bn, ws); */                                \
  } while (0)

#define TOOM22_MUL_REC_STRIDED(p, ps, a, an, as, b, bn, bs, ws, wss)                                \
  do {                                                                        \
    if (! MAYBE_mul_toom22                                                \
        || BELOW_THRESHOLD (bn, MUL_TOOM22_THRESHOLD))                        \
      PREFIX(mpn_mul_basecase_strided) (p, ps, a, an, as, b, bn, bs);                                \
    else if (4 * an < 5 * bn)                                                \
      PREFIX(mpn_toom22_mul_strided) (p, ps, a, an, as, b, bn, bs, ws, wss);                                \
    else                                                                \
      UNIMPLEMENTED; /* mpn_toom32_mul (p, a, an, b, bn, ws); */                                \
  } while (0)

__all__ void
PREFIX(mpn_toom22_mul) (mp_ptr pp,
                mp_srcptr ap, mp_size_t an,
                mp_srcptr bp, mp_size_t bn,
                mp_ptr scratch)
{
  mp_size_t n, s, t;
  int vm1_neg;
  mp_limb_t cy, cy2;
  mp_ptr asm1;
  mp_ptr bsm1;

#define a0  ap
#define a1  (ap + n)
#define b0  bp
#define b1  (bp + n)

  s = an >> 1;
  n = an - s;
  t = bn - n;

  ASSERT (an >= bn);

  ASSERT (0 < s && s <= n && s >= n - 1);
  ASSERT (0 < t && t <= s);

  asm1 = pp;
  bsm1 = pp + n;

  vm1_neg = 0;

  /* Compute asm1.  */
  if (s == n)
    {
      if (PREFIX(mpn_cmp) (a0, a1, n) < 0)
        {
          PREFIX(mpn_sub_n) (asm1, a1, a0, n);
          vm1_neg = 1;
        }
      else
        {
          PREFIX(mpn_sub_n) (asm1, a0, a1, n);
        }
    }
  else /* n - s == 1 */
    {
      if (a0[s] == 0 && PREFIX(mpn_cmp) (a0, a1, s) < 0)
        {
          PREFIX(mpn_sub_n) (asm1, a1, a0, s);
          asm1[s] = 0;
          vm1_neg = 1;
        }
      else
        {
          asm1[s] = a0[s] - PREFIX(mpn_sub_n) (asm1, a0, a1, s);
        }
    }

  /* Compute bsm1.  */
  if (t == n)
    {
      if (PREFIX(mpn_cmp) (b0, b1, n) < 0)
        {
          PREFIX(mpn_sub_n) (bsm1, b1, b0, n);
          vm1_neg ^= 1;
        }
      else
        {
          PREFIX(mpn_sub_n) (bsm1, b0, b1, n);
        }
    }
  else
    {
      if (PREFIX(mpn_zero_p) (b0 + t, n - t) && PREFIX(mpn_cmp) (b0, b1, t) < 0)
        {
          PREFIX(mpn_sub_n) (bsm1, b1, b0, t);
          MPN_ZERO (bsm1 + t, n - t);
          vm1_neg ^= 1;
        }
      else
        {
          PREFIX(mpn_sub) (bsm1, b0, n, b1, t);
        }
    }

#define v0        pp                                /* 2n */
#define vinf        (pp + 2 * n)                        /* s+t */
#define vm1        scratch                                /* 2n */
#define scratch_out        scratch + 2 * n

  /* vm1, 2n limbs */
  TOOM22_MUL_N_REC (vm1, asm1, bsm1, n, scratch_out);

  if (s > t)  TOOM22_MUL_REC (vinf, a1, s, b1, t, scratch_out);
  else        TOOM22_MUL_N_REC (vinf, a1, b1, s, scratch_out);

  /* v0, 2n limbs */
  TOOM22_MUL_N_REC (v0, ap, bp, n, scratch_out);

  /* H(v0) + L(vinf) */
  cy = PREFIX(mpn_add_n) (pp + 2 * n, v0 + n, vinf, n);

  /* L(v0) + H(v0) */
  cy2 = cy + PREFIX(mpn_add_n) (pp + n, pp + 2 * n, v0, n);

  /* L(vinf) + H(vinf) */
  cy += PREFIX(mpn_add) (pp + 2 * n, pp + 2 * n, n, vinf + n, s + t - n);

  if (vm1_neg)
    cy += PREFIX(mpn_add_n) (pp + n, pp + n, vm1, 2 * n);
  else
    cy -= PREFIX(mpn_sub_n) (pp + n, pp + n, vm1, 2 * n);

  ASSERT (cy + 1  <= 3);
  ASSERT (cy2 <= 2);

  MPN_INCR_U_VAR (pp + 2 * n, s + t, cy2);
  if (LIKELY (cy <= 2))
    /* if s+t==n, cy is zero, but we should not acces pp[3*n] at all. */
    MPN_INCR_U_VAR (pp + 3 * n, s + t - n, cy);
  else
    MPN_DECR_U_1 (pp + 3 * n, s + t - n);

#undef a0
#undef a1
#undef b0
#undef b1

#undef v0
#undef vinf
#undef vm1
#undef scratch_out
}

__all__ void
PREFIX(mpn_toom22_mul_strided) (mp_ptr pp, mp_size_t ps,
                mp_srcptr ap, mp_size_t an, mp_size_t as,
                mp_srcptr bp, mp_size_t bn, mp_size_t bs,
                mp_ptr scratch, mp_size_t ss)
{
  mp_size_t n, s, t;
  int vm1_neg;
  mp_limb_t cy, cy2;
  mp_ptr asm1;
  mp_ptr bsm1;

#define a0  ap
#define a1  (ap + n * as)
#define b0  bp
#define b1  (bp + n * bs)

  s = an >> 1;
  n = an - s;
  t = bn - n;

  ASSERT (an >= bn);

  ASSERT (0 < s && s <= n && s >= n - 1);
  ASSERT (0 < t && t <= s);

  asm1 = pp;
  bsm1 = pp + n * ps;

  vm1_neg = 0;

  /* Compute asm1.  */
  if (s == n)
    {
      if (PREFIX(mpn_cmp_strided) (a0, as, a1, as, n) < 0)
        {
          PREFIX(mpn_sub_n_strided) (asm1, ps, a1, as, a0, as, n);
          vm1_neg = 1;
        }
      else
        {
          PREFIX(mpn_sub_n_strided) (asm1, ps, a0, as, a1, as, n);
        }
    }
  else /* n - s == 1 */
    {
      if (a0[s * as] == 0 && PREFIX(mpn_cmp_strided) (a0, as, a1, as, s) < 0)
        {
          PREFIX(mpn_sub_n_strided) (asm1, ps, a1, as, a0, as, s);
          asm1[s * ps] = 0;
          vm1_neg = 1;
        }
      else
        {
          asm1[s * ps] = a0[s * as] - PREFIX(mpn_sub_n_strided) (asm1, ps, a0, as, a1, as, s);
        }
    }

  /* Compute bsm1.  */
  if (t == n)
    {
      if (PREFIX(mpn_cmp_strided) (b0, bs, b1, bs, n) < 0)
        {
          PREFIX(mpn_sub_n_strided) (bsm1, ps, b1, bs, b0, bs, n);
          vm1_neg ^= 1;
        }
      else
        {
          PREFIX(mpn_sub_n_strided) (bsm1, ps, b0, bs, b1, bs, n);
        }
    }
  else
    {
      if (PREFIX(mpn_zero_p_strided) (b0 + t, n - t, bs) && PREFIX(mpn_cmp_strided) (b0, bs, b1, bs, t) < 0)
        {
          PREFIX(mpn_sub_n_strided) (bsm1, ps, b1, bs, b0, bs, t);
          MPN_ZERO_STRIDED (bsm1 + t, n - t, ps);
          vm1_neg ^= 1;
        }
      else
        {
          PREFIX(mpn_sub_strided) (bsm1, ps, b0, n, bs, b1, t, bs);
        }
    }

#define v0        pp                                /* 2n */
#define vinf        (pp + 2 * n * ps)                        /* s+t */
#define vm1        scratch                                /* 2n */
#define scratch_out        scratch + 2 * n * ss

  /* vm1, 2n limbs */
  TOOM22_MUL_N_REC_STRIDED (vm1, ss, asm1, ps, bsm1, ps, n, scratch_out, ss);

  if (s > t)  TOOM22_MUL_REC_STRIDED (vinf, ss, a1, s, as, b1, t, bs, scratch_out, ss);
  else        TOOM22_MUL_N_REC_STRIDED (vinf, ps, a1, as, b1, bs, s, scratch_out, ss);

  /* v0, 2n limbs */
  TOOM22_MUL_N_REC_STRIDED (v0, ps, ap, as, bp, bs, n, scratch_out, ss);

  /* H(v0) + L(vinf) */
  cy = PREFIX(mpn_add_n_strided) (pp + 2 * n * ps, ps, v0 + n * ps, ps, vinf, ps, n);

  /* L(v0) + H(v0) */
  cy2 = cy + PREFIX(mpn_add_n_strided) (pp + n * ps, ps, pp + 2 * n * ps, ps, v0, ps, n);

  /* L(vinf) + H(vinf) */
  cy += PREFIX(mpn_add_strided) (pp + 2 * n * ps, ps, pp + 2 * n * ps, n, ps, vinf + n * ps, s + t - n, ps);

  if (vm1_neg)
    cy += PREFIX(mpn_add_n_strided) (pp + n * ps, ps, pp + n * ps, ps, vm1, ss, 2 * n);
  else
    cy -= PREFIX(mpn_sub_n_strided) (pp + n * ps, ps, pp + n * ps, ps, vm1, ss, 2 * n);

  ASSERT (cy + 1  <= 3);
  ASSERT (cy2 <= 2);

  MPN_INCR_U_VAR_STRIDED (pp + 2 * n * ps, s + t, ps, cy2);
  if (LIKELY (cy <= 2))
    /* if s+t==n, cy is zero, but we should not acces pp[3*n] at all. */
    MPN_INCR_U_VAR_STRIDED (pp + 3 * n * ps, s + t - n, ps, cy);
  else
    MPN_DECR_U_1_STRIDED (pp + 3 * n * ps, s + t - n, ps);
}

// vim: set sw=2 ts=8 noet:
