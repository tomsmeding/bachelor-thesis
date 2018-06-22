#pragma once

#include "gmp-es-mpz.h"


void gpu_crt_split(struct PREFIX(__mpz_t) *m, struct PREFIX(__mpz_t) *n, struct PREFIX(__mpz_t) *mf, struct PREFIX(__mpz_t) *nf, const struct PREFIX(__mpz_t) *x, const struct PREFIX(__mpz_t) **list, PREFIX(mp_size_t) listlen);

void gpu_crt_merge(struct PREFIX(__mpz_t) *x, const struct PREFIX(__mpz_t) *a, const struct PREFIX(__mpz_t) *b, const struct PREFIX(__mpz_t) *m, const struct PREFIX(__mpz_t) *n, const struct PREFIX(__mpz_t) *mf, const struct PREFIX(__mpz_t) *nf);

int gpu_crt_make_split_tree(PREFIX(mpz_t) **msp, PREFIX(mpz_t) **nsp, PREFIX(mpz_t) **mfsp, PREFIX(mpz_t) **nfsp, const struct PREFIX(__mpz_t) *x, PREFIX(mp_limb_t) limit);

void gpu_crt_split_list(PREFIX(mpz_t) *res, const struct PREFIX(__mpz_t) *x, const PREFIX(mpz_t) *ms, const PREFIX(mpz_t) *ns, int length);

void gpu_crt_merge_list(struct PREFIX(__mpz_t) *x, const PREFIX(mpz_t) *split, const PREFIX(mpz_t) *ms, const PREFIX(mpz_t) *ns, const PREFIX(mpz_t) *mfs, const PREFIX(mpz_t) *nfs, int length);

// vim: set sw=4 ts=4 noet:
