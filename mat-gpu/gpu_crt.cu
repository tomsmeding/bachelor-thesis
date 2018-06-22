#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include "gpu_crt.h"
#include "crt.h"

static inline bool is_little_endian(void) {
	union {
		uint32_t i;
		char c[4];
	} e = { 0x01000000 };

	return e.c[0] == 1;
}

// TODO convert const numbers by re-pointing pointers instead of memory allocation

// Do not initialise z beforehand!
static void gmp_from_gpu(mpz_t z, const gpu_mpz_t w) {
	z->_mp_size = (w->size * sizeof(gpu_mp_limb_t) + sizeof(mp_limb_t) - 1) / sizeof(mp_limb_t);
	z->_mp_alloc = z->_mp_size + 2;
	z->_mp_d = (mp_limb_t*)malloc(z->_mp_alloc * sizeof(mp_limb_t));

	assert(sizeof(mp_limb_t) / sizeof(gpu_mp_limb_t) <= 2);
	assert(sizeof(gpu_mp_limb_t) / sizeof(mp_limb_t) <= 2);
	assert(is_little_endian());

	z->_mp_d[z->_mp_size - 1] = 0;
	memcpy(z->_mp_d, w->ptr, w->size * sizeof(gpu_mp_limb_t));
	while (z->_mp_size > 0 && z->_mp_d[z->_mp_size - 1] == 0) z->_mp_size--;
}

static void gpu_from_gmp(gpu_mpz_t z, const mpz_t w) {
	z->size = (w->_mp_size * sizeof(mp_limb_t) + sizeof(gpu_mp_limb_t) - 1) / sizeof(gpu_mp_limb_t);
	assert(z->size <= z->cap);

	assert(sizeof(mp_limb_t) / sizeof(gpu_mp_limb_t) <= 2);
	assert(sizeof(gpu_mp_limb_t) / sizeof(mp_limb_t) <= 2);
	assert(is_little_endian());

	z->ptr[z->size - 1] = 0;
	memcpy(z->ptr, w->_mp_d, w->_mp_size * sizeof(mp_limb_t));
	while (z->size > 0 && z->ptr[z->size - 1] == 0) z->size--;

	free(w->_mp_d);
}


void gpu_crt_split(struct gpu___mpz_t *m, struct gpu___mpz_t *n, struct gpu___mpz_t *mf, struct gpu___mpz_t *nf, const struct gpu___mpz_t *x, const struct gpu___mpz_t **list, gpu_mp_size_t listlen) {
	mpz_t gmp_m, gmp_n, gmp_mf, gmp_nf, gmp_x, *gmp_list_data;
	mpz_srcptr *gmp_list;

	gmp_from_gpu(gmp_m, m);
	gmp_from_gpu(gmp_n, n);
	gmp_from_gpu(gmp_mf, mf);
	gmp_from_gpu(gmp_nf, nf);
	gmp_from_gpu(gmp_x, x);

	gmp_list_data = (mpz_t*)malloc(listlen * sizeof(mpz_t));
	gmp_list = (mpz_srcptr*)malloc(listlen * sizeof(mpz_srcptr));
	for (int i = 0; i < listlen; i++) {
		gmp_list[i] = gmp_list_data[i];
		gmp_from_gpu(gmp_list_data[i], list[i]);
	}

	crt_split(gmp_m, gmp_n, gmp_mf, gmp_nf, gmp_x, gmp_list, listlen);

	gpu_from_gmp(m, gmp_m);
	gpu_from_gmp(n, gmp_n);
	gpu_from_gmp(mf, gmp_mf);
	gpu_from_gmp(nf, gmp_nf);

	for (int i = 0; i < listlen; i++) free(gmp_list_data[i]->_mp_d);

	free(gmp_list);
	free(gmp_list_data);
}

void gpu_crt_merge(struct gpu___mpz_t *x, const struct gpu___mpz_t *a, const struct gpu___mpz_t *b, const struct gpu___mpz_t *m, const struct gpu___mpz_t *n, const struct gpu___mpz_t *mf, const struct gpu___mpz_t *nf) {
	mpz_t gmp_x, gmp_a, gmp_b, gmp_m, gmp_n, gmp_mf, gmp_nf;

	gmp_from_gpu(gmp_x, x);
	gmp_from_gpu(gmp_a, a);
	gmp_from_gpu(gmp_b, b);
	gmp_from_gpu(gmp_m, m);
	gmp_from_gpu(gmp_n, n);
	gmp_from_gpu(gmp_mf, mf);
	gmp_from_gpu(gmp_nf, nf);

	crt_merge(gmp_x, gmp_a, gmp_b, gmp_m, gmp_n, gmp_mf, gmp_nf);

	gpu_from_gmp(x, gmp_x);
	mpz_clears(gmp_a, gmp_b, gmp_m, gmp_n, gmp_mf, gmp_nf, NULL);
}

int gpu_crt_make_split_tree(gpu_mpz_t **msp, gpu_mpz_t **nsp, gpu_mpz_t **mfsp, gpu_mpz_t **nfsp, const struct gpu___mpz_t *x, gpu_mp_limb_t limit) {
	mpz_t *gmp_ms, *gmp_ns, *gmp_mfs, *gmp_nfs, gmp_x;

	gmp_from_gpu(gmp_x, x);

	int length = crt_make_split_tree(&gmp_ms, &gmp_ns, &gmp_mfs, &gmp_nfs, gmp_x, limit);

	*msp = (gpu_mpz_t*)malloc(length * sizeof(gpu_mpz_t));
	*nsp = (gpu_mpz_t*)malloc(length * sizeof(gpu_mpz_t));
	*mfsp = (gpu_mpz_t*)malloc(length * sizeof(gpu_mpz_t));
	*nfsp = (gpu_mpz_t*)malloc(length * sizeof(gpu_mpz_t));

	for (int i = 0; i < length; i++) {
		gpu_from_gmp((*msp)[i], gmp_ms[i]);
		gpu_from_gmp((*nsp)[i], gmp_ns[i]);
		gpu_from_gmp((*mfsp)[i], gmp_mfs[i]);
		gpu_from_gmp((*nfsp)[i], gmp_nfs[i]);
	}
	free(gmp_ms);
	free(gmp_ns);
	free(gmp_mfs);
	free(gmp_nfs);

	return length;
}

void gpu_crt_split_list(gpu_mpz_t *res, const struct gpu___mpz_t *x, const gpu_mpz_t *ms, const gpu_mpz_t *ns, int length) {
	mpz_t *gmp_res, gmp_x, *gmp_ms, *gmp_ns;

	gmp_ms = (mpz_t*)malloc(length * sizeof(mpz_t));
	gmp_ns = (mpz_t*)malloc(length * sizeof(mpz_t));

	for (int i = 0; i < length; i++) {
		gmp_from_gpu(gmp_ms[i], ms[i]);
		gmp_from_gpu(gmp_ns[i], ns[i]);
	}

	gmp_from_gpu(gmp_x, x);

	gmp_res = (mpz_t*)malloc((1 << length) * sizeof(mpz_t));

	crt_split_list(gmp_res, gmp_x, gmp_ms, gmp_ns, length);

	for (int i = 0; i < (1 << length); i++) {
		gpu_from_gmp(res[i], gmp_res[i]);
	}
	free(gmp_res);

	mpz_clear(gmp_x);
	for (int i = 0; i < length; i++) {
		mpz_clear(gmp_ms[i]);
		mpz_clear(gmp_ns[i]);
	}
	free(gmp_ms);
	free(gmp_ns);
}

void gpu_crt_merge_list(struct gpu___mpz_t *x, const gpu_mpz_t *split, const gpu_mpz_t *ms, const gpu_mpz_t *ns, const gpu_mpz_t *mfs, const gpu_mpz_t *nfs, int length) {
	mpz_t gmp_x, *gmp_split, *gmp_ms, *gmp_ns, *gmp_mfs, *gmp_nfs;

	mpz_init(gmp_x);

	gmp_split = (mpz_t*)malloc((1 << length) * sizeof(mpz_t));
	gmp_ms = (mpz_t*)malloc(length * sizeof(mpz_t));
	gmp_ns = (mpz_t*)malloc(length * sizeof(mpz_t));
	gmp_mfs = (mpz_t*)malloc(length * sizeof(mpz_t));
	gmp_nfs = (mpz_t*)malloc(length * sizeof(mpz_t));

	for (int i = 0; i < (1 << length); i++) {
		gmp_from_gpu(gmp_split[i], split[i]);
	}

	for (int i = 0; i < length; i++) {
		gmp_from_gpu(gmp_ms[i], ms[i]);
		gmp_from_gpu(gmp_ns[i], ns[i]);
		gmp_from_gpu(gmp_mfs[i], mfs[i]);
		gmp_from_gpu(gmp_nfs[i], nfs[i]);
	}

	crt_merge_list(gmp_x, gmp_split, gmp_ms, gmp_ns, gmp_mfs, gmp_nfs, length);

	gpu_from_gmp(x, gmp_x);

	for (int i = 0; i < (1 << length); i++) {
		mpz_clear(gmp_split[i]);
	}
	free(gmp_split);

	for (int i = 0; i < length; i++) {
		mpz_clear(gmp_ms[i]);
		mpz_clear(gmp_ns[i]);
		mpz_clear(gmp_mfs[i]);
		mpz_clear(gmp_nfs[i]);
	}
	free(gmp_ms);
	free(gmp_ns);
	free(gmp_mfs);
	free(gmp_nfs);
}

// vim: set sw=4 ts=4 noet:
