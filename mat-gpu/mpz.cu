#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>
#include <assert.h>
#include "gmp-es-mpz.h"
#include "gmp-es-internal-convert.h"


void PREFIX(mpz_init)(mpz_t z) {
	z->size = 0;
}

void PREFIX(mpz_clear)(mpz_t z) {}

void PREFIX(mpz_set)(mpz_t d, const mpz_t s) {
	d->size = s->size;
	memcpy(d->ptr, s->ptr, s->size * sizeof(mp_limb_t));
}

void PREFIX(gmp_randinit_default)(gmp_randstate_t rs) {
	FILE *f = fopen("/dev/urandom", "r");
	unsigned int seed;
	fread(rs->state, 1, sizeof rs->state, f);
	fread(&seed, 1, sizeof seed, f);
	fclose(f);

	memset(&rs->d, 0, sizeof rs->d);
	initstate_r(seed, rs->state, sizeof rs->state, &rs->d);
}

void PREFIX(gmp_randinit_seed)(gmp_randstate_t rs, uint64_t seed) {
	static const uint64_t magic = 0x05fb6911ba54b4ed;

	uint64_t val = seed ^ magic;
	memcpy(rs->state, (void*)&val, 8);
	memset(rs->state + 8, 0, sizeof rs->state - 8);

	memset(&rs->d, 0, sizeof rs->d);
	initstate_r((unsigned)seed, rs->state, sizeof rs->state, &rs->d);
}

static mp_limb_t random_limb(gmp_randstate_t rs) {
	const mp_limb_t rm = RAND_MAX;
	assert(((rm + 1) & rm) == 0 && "RAND_MAX+1 should be a power of 2");

	mp_limb_t r = 0;
	mp_limb_t n = rm + 1;
	do {
		int32_t val;
		random_r(&rs->d, &val);
		r = RAND_MAX * r + (mp_limb_t)val;
		n *= rm + 1;
	} while (n);

	return r;
}

void PREFIX(mpz_urandomb)(mpz_t z, gmp_randstate_t rs, int nbits) {
	const int limbs_required = (nbits + GPU_GMP_LIMB_BITS - 1) / GPU_GMP_LIMB_BITS;

	assert(limbs_required <= z->cap);

	z->size = limbs_required;
	int irregular = nbits % GPU_GMP_LIMB_BITS != 0;
	for (int i = 0; i < limbs_required - irregular; i++) {
		z->ptr[i] = random_limb(rs);
	}
	if (irregular) {
		z->ptr[limbs_required - 1] = random_limb(rs) & ((1 << (nbits % GPU_GMP_LIMB_BITS)) - 1);
	}
}

mp_size_t PREFIX(gmp_urandomb_ui)(gmp_randstate_t rs, int nbits) {
	assert(nbits <= 8 * sizeof(mp_size_t));
	assert(sizeof(mp_size_t) <= sizeof(mp_limb_t));
	mp_size_t v = random_limb(rs);
	if (nbits == 8 * sizeof(mp_size_t)) return v;
	else return v & ((1 << nbits) - 1);
}

void PREFIX(mpz_out_str_16)(FILE *f, const mpz_t z) {
	if (z->size == 0) {
		fprintf(f, "0");
		return;
	}
	mp_size_t size = z->size;
	if (size < -1 || (size == -1 && z->ptr[0] != 0)) {
		fputc('-', f);
		size = -size;
	}
	bool print = false;
	for (int i = size - 1; i >= 0; i--) {
		for (int k = 2 * GPU_GMP_LIMB_BYTES - 1; k >= 0; k--) {
			char c = "0123456789abcdef"[(z->ptr[i] >> (4 * k)) & 0xf];
			if (c != '0') print = true;
			if (print) fputc(c, f);
		}
	}
	if (!print) fputc('0', f);
}

static mp_size_t inp_buffer_hex(mp_ptr ptr, const unsigned char *buf, mp_size_t numchars) {
	mp_size_t j = 0, k = 0;
	mp_limb_t v = 0;
	for (mp_size_t i = numchars - 1; i >= 0; i -= 2) {
		v |= (buf[i] + (i > 0 ? buf[i-1] << 4 : 0)) << (8 * k++);
		if (k == GPU_GMP_LIMB_BYTES || i < 2) {
			ptr[j++] = v;
			k = 0;
			v = 0;
		}
	}
	return j;
}

void PREFIX(mpz_inp_str)(mpz_t z, const char *str) {
	const mp_size_t buflen = z->cap * 2 * GPU_GMP_LIMB_BYTES;
	unsigned char buf[buflen];

	while (isspace(str[0])) str++;
	bool neg = false;
	if (str[0] == '-') { neg = true; str++; }
	if (str[0] == '0' && str[1] == 'x') str += 2;

	mp_size_t numchars = 0;
	for (numchars = 0; numchars < buflen; numchars++) {
		if (str[numchars] == '\0') break;
		char c = tolower(str[numchars]);
		if ('0' <= c && c <= '9') c -= '0';
		else if ('a' <= c && c <= 'f') c -= 'a' - 10;
		else {
			fprintf(stderr, "Invalid character '%c' in mpz_inp_str\n", str[numchars]);
			assert(false);
		}
		buf[numchars] = c;
	}
	assert(str[numchars] == '\0');

	z->size = inp_buffer_hex(z->ptr, buf, numchars);

	if (neg) gpu_mpz_neg(z, z);
}

void PREFIX(mpz_inp_file)(mpz_t z, FILE *f) {
	const mp_size_t buflen = z->cap * 2 * GPU_GMP_LIMB_BYTES;
	unsigned char buf[buflen];

	bool neg = false;
	mp_size_t numchars = 0;
	for (numchars = 0; numchars < buflen + 1; numchars++) {
		char c = tolower(fgetc(f));
		if (isspace(c) && numchars == 0) {
			numchars--;
			continue;
		}
		if (c == '-' && numchars == 0) {
			neg = true;
			numchars--;
			continue;
		}
		if (!isdigit(c) && (c < 'a' || c > 'f')) {
			ungetc(c, f);
			break;
		} else {
			assert(numchars < buflen);
			if ('0' <= c && c <= '9') c -= '0';
			else if ('a' <= c && c <= 'f') c -= 'a' - 10;
			else assert(false);
			buf[numchars] = c;
		}
	}

	z->size = inp_buffer_hex(z->ptr, buf, numchars);

	if (neg) gpu_mpz_neg(z, z);
}

static void out_file_split(FILE *f, const mp_limb_t *ptr, mp_size_t nlimbs) {
	if (nlimbs == 0) {
		fprintf(f, "0");
		return;
	}
	if (nlimbs < -1 || (nlimbs == -1 && ptr[0] != 0)) {
		fprintf(f, "-");
		nlimbs = -nlimbs;
	}
	bool print = false;
	for (mp_size_t i = nlimbs - 1; i >= 0; i--) {
		for (mp_size_t k = 2 * GPU_GMP_LIMB_BYTES - 1; k >= 0; k--) {
			char c = "0123456789abcdef"[(ptr[i] >> (4 * k)) & 0xf];
			if (c != '0') print = true;
			if (print) fputc(c, f);
		}
	}
	if (!print) fputc('0', f);
}

void PREFIX(mpz_out_file)(FILE *f, const mpz_t z) {
	out_file_split(f, z->ptr, z->size);
}

// vim: set sw=4 ts=4 noet:
