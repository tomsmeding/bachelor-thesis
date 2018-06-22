#pragma once

#include "gmp.h"


// Finds two numbers m and n, both close to sqrt(x), such that m*n > x and m
// and n are coprime to each other and each of the numbers in 'list' (of
// 'listlen' items).
// mf and nf get factors such that mf * m + nf * n = 1; these can be used to
// reconstruct x from x % m, x % n, m and n.
// Reduction of x modulo m and n gives a ring isomorphism between Z/mnZ and
// (Z/mZ) x (Z/nZ).
// The function ensures that m < n.
// Overlap: Only between const inputs.
void crt_split(mpz_ptr m, mpz_ptr n, mpz_ptr mf, mpz_ptr nf, mpz_srcptr x, mpz_srcptr *list, mp_size_t listlen);

// Inverse of crt_split.
// Overlap: Only x == a allowed, and separately between const inputs.
void crt_merge(mpz_ptr x, mpz_srcptr a, mpz_srcptr b, mpz_srcptr m, mpz_srcptr n, mpz_srcptr mf, mpz_srcptr nf);

// Allocates arrays ms and ns and writes pairs of numbers such that when x is
// reduced modulo one of each pair for each pair in the list, it will end up
// below limit. Using the corresponding numbers in mfs and nfs, and supposing
// one has reduced x modulo both m and n each time, x may be reconstructed
// using crt_merge.
// Returns length of allocated lists.
// Overlap: Only between const inputs.
int crt_make_split_tree(mpz_t **msp, mpz_t **nsp, mpz_t **mfsp, mpz_t **nfsp, mpz_srcptr x, mp_limb_t limit);

// Assumes 'res' holds 2^length initialised elements. Reduces in a binary tree
// modulo the pairs (m, n) and writes the leaves to 'res'. Use lists from
// crt_make_split_tree.
// Overlap: Only between const inputs, not that that's useful.
void crt_split_list(mpz_t *res, mpz_srcptr x, const mpz_t *ms, const mpz_t *ns, int length);

// Inverse of crt_split_list.
// Overlap: Only between const inputs, not that that's useful.
void crt_merge_list(mpz_ptr x, const mpz_t *split, const mpz_t *ms, const mpz_t *ns, const mpz_t *mfs, const mpz_t *nfs, int length);

// vim: set sw=4 ts=4 noet:
