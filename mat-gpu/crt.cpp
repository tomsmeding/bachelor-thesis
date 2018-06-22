#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include "crt.h"


// Finds two numbers m and n, both close to sqrt(x), such that m*n > x and m
// and n are coprime to each other and each of the numbers in 'list' (of
// 'listlen' items).
// mf and nf get factors such that mf * m + nf * n = 1; these can be used to
// reconstruct x from x%m, x%n, m and n.
// Reduction of x modulo m and n gives a ring isomorphism between Z/mnZ and
// (Z/mZ) x (Z/nZ).
// The function ensures that m < n.
// Overlap: None allowed.
void crt_split(mpz_ptr m, mpz_ptr n, mpz_ptr mf, mpz_ptr nf, mpz_srcptr x, mpz_srcptr *list, mp_size_t listlen) {
	mpz_t temp;
	mpz_init(temp);

	mpz_sqrt(m, x);
	while (true) {
		bool success = true;
		for (mp_size_t i = 0; i < listlen; i++) {
			mpz_gcd(temp, m, list[i]);
			if (mpz_cmp_ui(temp, 1) != 0) {
				success = false;
				break;
			}
		}
		if (success) break;

		mpz_add_ui(m, m, 1);
	}

	// m >= mpz_sqrt(x) >= sqrt(x) - 1
	// (sqrt(x) - 1) * (sqrt(x) + 2) = x + sqrt(x) - 2 >= x  if x >= 1, which we just assume it is
	// and sqrt(x) + 2 = (sqrt(x) - 1) + 3
	mpz_add_ui(n, m, 3);

	bool m_even = mpz_even_p(m);

	while (true) {
		mpz_gcdext(temp, mf, nf, m, n);
		if (mpz_cmp_ui(temp, 1) == 0) {
			bool success = true;
			for (mp_size_t i = 0; i < listlen; i++) {
				mpz_gcd(temp, n, list[i]);
				if (mpz_cmp_ui(temp, 1) != 0) {
					success = false;
					break;
				}
			}
			if (success) break;
		}

		// If m is even, we can't have gcd(m, n) = 1 if n is also even. Since
		// we took n = m + 3, they already have different parity, so this is a
		// valid optimisation.
		mpz_add_ui(n, n, m_even ? 2 : 1);
	}

	mpz_clear(temp);
}

// Overlap: Only x == a allowed.
void crt_merge(mpz_ptr x, mpz_srcptr a, mpz_srcptr b, mpz_srcptr m, mpz_srcptr n, mpz_srcptr mf, mpz_srcptr nf) {
	mpz_t temp;
	mpz_init(temp);

	mpz_mul(x, a, nf);
	mpz_mul(x, x, n);
	mpz_mul(temp, b, mf);
	mpz_addmul(x, temp, m);
	mpz_mul(temp, m, n);
	mpz_mod(x, x, temp);

	mpz_clear(temp);
}

static int ilog2(mp_limb_t n) {
	return 8 * sizeof n - __builtin_clzll(n) - 1;
}

// Allocates arrays ms and ns and writes pairs of numbers such that when x is
// reduced modulo one of each pair for each pair in the list, it will end up
// below limit. Using the corresponding numbers in mfs and nfs, and supposing
// one has reduced x modulo both m and n each time, x may be reconstructed
// using crt_merge.
// Returns length of allocated lists.
// Overlap: None allowed.
int crt_make_split_tree(mpz_t **msp, mpz_t **nsp, mpz_t **mfsp, mpz_t **nfsp, mpz_srcptr x_, mp_limb_t limit) {
	mpz_t x;
	mpz_init_set(x, x_);

	int cap = mpz_sizeinbase(x, 2) / ilog2(limit) + 1;
	mpz_t *ms = (mpz_t*)malloc(cap * sizeof(mpz_t));
	mpz_t *ns = (mpz_t*)malloc(cap * sizeof(mpz_t));
	mpz_t *mfs = (mpz_t*)malloc(cap * sizeof(mpz_t));
	mpz_t *nfs = (mpz_t*)malloc(cap * sizeof(mpz_t));

	mpz_srcptr *list = (mpz_srcptr*)malloc(2 * cap * sizeof(mpz_srcptr));

	int num;
	for (num = 0; mpz_cmp_ui(x, limit) >= 0; num++) {
		assert(num < cap);

		mpz_inits(ms[num], ns[num], mfs[num], nfs[num], NULL);
		crt_split(ms[num], ns[num], mfs[num], nfs[num], x, list, 2 * num);
		list[2 * num] = &ms[num][0];
		list[2 * num + 1] = &ns[num][0];

		mpz_mod(x, x, ns[num]);
	}

	free(list);
	mpz_clear(x);

	*msp = ms; *nsp = ns; *mfsp = mfs; *nfsp = nfs;
	return num;
}

// Assumes 'res' holds 2^length initialised elements. Reduces in a binary tree
// modulo the pairs (m, n) and writes the leaves to 'res'. Use lists from
// crt_make_split_tree.
// Overlap: None allowed.
void crt_split_list(mpz_t *res, mpz_srcptr x, const mpz_t *ms, const mpz_t *ns, int length) {
	if (length <= 0) {
		mpz_set(res[0], x);
		return;
	}

	mpz_t y;
	mpz_init(y);
	mpz_mod(y, x, ms[0]);
	crt_split_list(res, y, ms + 1, ns + 1, length - 1);

	res += 1 << (length - 1);

	mpz_mod(y, x, ns[0]);
	crt_split_list(res, y, ms + 1, ns + 1, length - 1);
	mpz_clear(y);
}

// Overlap: None allowed.
void crt_merge_list(mpz_ptr x, const mpz_t *split, const mpz_t *ms, const mpz_t *ns, const mpz_t *mfs, const mpz_t *nfs, int length) {
	if (length <= 0) {
		mpz_set(x, split[0]);
		return;
	}
	if (length == 1) {
		crt_merge(x, split[0], split[1], ms[0], ns[0], mfs[0], nfs[0]);
		return;
	}

	mpz_t y;
	mpz_init(y);

	crt_merge_list(x, split, ms + 1, ns + 1, mfs + 1, nfs + 1, length - 1);
	split += 1 << (length - 1);
	crt_merge_list(y, split, ms + 1, ns + 1, mfs + 1, nfs + 1, length - 1);
	crt_merge(x, x, y, ms[0], ns[0], mfs[0], nfs[0]);

	mpz_clear(y);
}

// vim: set sw=4 ts=4 noet:
