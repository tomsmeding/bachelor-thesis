#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gmp-es.h"

// Apparently, on an NVIDIA GPU the L1 cache line size is 128 bytes and the L2
// line size is 32 bytes. Making sizeof(__mpz_t) a multiple of 128 bytes sounds
// like overkill, but making it a multiple of 32 bytes is certainly doable.
// Stupid benchmark: Strassen with 512x512 matrices of 256 bit ints with chunk
// size 256 gives:
// - NUM_SIZE = 512+6*32: 1.63 seconds
// - NUM_SIZE = 512+7*32: 1.54 seconds
// Note that (512+7*32)/8 + sizeof(mp_size_t) = 96 = 3 * 32, so the second test
// is aligned while the first test isn't.

#define NUM_SIZE (4096+7*32)
// #define NUM_SIZE (512+7*32)


struct PREFIX(__mpz_t) {
	static const PREFIX(mp_size_t) cap = NUM_SIZE / GPU_GMP_LIMB_BITS;

	PREFIX(mp_limb_t) ptr[cap];
	PREFIX(mp_size_t) size;
};

typedef struct PREFIX(__mpz_t) PREFIX(mpz_t)[1];

struct PREFIX(__gmp_randstate_t) {
	struct random_data d;
	char state[64];
};

typedef struct PREFIX(__gmp_randstate_t) PREFIX(gmp_randstate_t)[1];

#define STRIDED_PTR(p__) ((PREFIX(mp_limb_t)*)(p__))
#define STRIDED_SIZE(p__, str__) (((PREFIX(mp_size_t)*)(p__))[PREFIX(__mpz_t)::cap * (str__)])
#define STRIDED_ALLOC(p__) (PREFIX(__mpz_t)::cap)

#define PTR_SHIFT(p__, amt__) ((struct PREFIX(__mpz_t)*)((uint8_t*)(p__) + (amt__) * sizeof(PREFIX(mp_limb_t))))

void PREFIX(mpz_init)(PREFIX(mpz_t) z);
void PREFIX(mpz_clear)(PREFIX(mpz_t) z);
void PREFIX(mpz_set)(PREFIX(mpz_t) d, const PREFIX(mpz_t) s);
// __all__ void PREFIX(mpz_addmul)(PREFIX(mpz_t) d, const PREFIX(mpz_t) a, const PREFIX(mpz_t) b, PREFIX(mpz_t) scratch);
// __all__ void PREFIX(mpz_addmul_base)(PREFIX(mpz_t) d, const PREFIX(mpz_t) a, const PREFIX(mpz_t) b, PREFIX(mpz_t) scratch);
// __all__ void PREFIX(mpz_mul)(PREFIX(mpz_t) d, const PREFIX(mpz_t) a, const PREFIX(mpz_t) b);
// __all__ void PREFIX(mpz_mul_base)(PREFIX(mpz_t) d, const PREFIX(mpz_t) a, const PREFIX(mpz_t) b);
// __all__ void PREFIX(mpz_add)(PREFIX(mpz_t) d, const PREFIX(mpz_t) a, const PREFIX(mpz_t) b);
// __all__ void PREFIX(mpz_sub)(PREFIX(mpz_t) d, const PREFIX(mpz_t) a, const PREFIX(mpz_t) b);

// __all__ void PREFIX(mpz_addmul_base_strided)(struct __PREFIX(mpz_t) *d, PREFIX(mp_size_t) ds, const struct __PREFIX(mpz_t) *a, PREFIX(mp_size_t) as, const struct __PREFIX(mpz_t) *b, PREFIX(mp_size_t) bs, struct __PREFIX(mpz_t) *scratch, PREFIX(mp_size_t) ss);
// __all__ void PREFIX(mpz_mul_base_strided)(struct __PREFIX(mpz_t) *d, PREFIX(mp_size_t) ds, const struct __PREFIX(mpz_t) *a, PREFIX(mp_size_t) as, const struct __PREFIX(mpz_t) *b, PREFIX(mp_size_t) bs);
// __all__ void PREFIX(mpz_add_strided)(struct __PREFIX(mpz_t) *d, PREFIX(mp_size_t) ds, const struct __PREFIX(mpz_t) *a, PREFIX(mp_size_t) as, const struct __PREFIX(mpz_t) *b, PREFIX(mp_size_t) bs);

void PREFIX(gmp_randinit_default)(PREFIX(gmp_randstate_t) rs);
void PREFIX(gmp_randinit_seed)(PREFIX(gmp_randstate_t) rs, uint64_t seed);
void PREFIX(mpz_urandomb)(PREFIX(mpz_t) z, PREFIX(gmp_randstate_t) rs, int nbits);
PREFIX(mp_size_t) PREFIX(gmp_urandomb_ui)(PREFIX(gmp_randstate_t) rs, int nbits);

void PREFIX(mpz_out_str_16)(FILE *f, const PREFIX(mpz_t) z);
void PREFIX(mpz_inp_str)(PREFIX(mpz_t) z, const char *str);
void PREFIX(mpz_inp_file)(PREFIX(mpz_t) z, FILE *f);
void PREFIX(mpz_out_file)(FILE *f, const PREFIX(mpz_t) z);


// --- ABRIDGED GMP MPZ IMPLEMENTATION ---

#define MPZ_REALLOC(z,n) (UNLIKELY ((n) > ALLOC(z)) ? PREFIX(_mpz_realloc) (z,n) : PTR(z))

#define MPZ_REALLOC_STRIDED(z,n) (UNLIKELY ((n) > STRIDED_ALLOC(z)) ? PREFIX(_mpz_realloc) (z,n) : STRIDED_PTR(z))

#define PTR(z) ((z)->ptr)
#define SIZ(z) ((z)->size)
#define ALLOC(z) ((z)->cap)

#define ABS(n) (abs(n))

typedef const struct PREFIX(__mpz_t) *PREFIX(mpz_srcptr);
typedef struct PREFIX(__mpz_t) *PREFIX(mpz_ptr);

#define MPZ_SRCPTR_SWAP(z,w) do { PREFIX(mpz_srcptr) t = z; z = w; w = t; } while (0)
#define MPZ_PTR_SWAP(z,w) do { PREFIX(mpz_ptr) t = z; z = w; w = t; } while (0)
#define MP_SIZE_T_SWAP(n,m) do { PREFIX(mp_size_t) t = n; n = m; m = t; } while (0)

#define MPN_NORMALIZE(DST, NLIMBS)       \
  do {                                   \
    while ((NLIMBS) > 0)                 \
      {                                  \
        if ((DST)[(NLIMBS) - 1] != 0)    \
          break;                         \
        (NLIMBS)--;                      \
      }                                  \
  } while (0)

#define MPN_NORMALIZE_STRIDED(DST, NLIMBS, STRIDE)       \
  do {                                   \
    while ((NLIMBS) > 0)                 \
      {                                  \
        if ((DST)[((NLIMBS) - 1) * (STRIDE)] != 0)    \
          break;                         \
        (NLIMBS)--;                      \
      }                                  \
  } while (0)

__all__ void PREFIX(mpz_add)(PREFIX(mpz_ptr) w, PREFIX(mpz_srcptr) u, PREFIX(mpz_srcptr) v);
__all__ void PREFIX(mpz_sub)(PREFIX(mpz_ptr) w, PREFIX(mpz_srcptr) u, PREFIX(mpz_srcptr) v);
__all__ void PREFIX(mpz_mul)(PREFIX(mpz_ptr) w, PREFIX(mpz_srcptr) u, PREFIX(mpz_srcptr) v);
__all__ void PREFIX(mpz_mul_base)(PREFIX(mpz_ptr) w, PREFIX(mpz_srcptr) u, PREFIX(mpz_srcptr) v);
__all__ void PREFIX(mpz_addmul) (PREFIX(mpz_ptr) w, PREFIX(mpz_srcptr) u, PREFIX(mpz_srcptr) v, PREFIX(mpz_ptr) scratch);
__all__ void PREFIX(mpz_addmul_base) (PREFIX(mpz_ptr) w, PREFIX(mpz_srcptr) u, PREFIX(mpz_srcptr) v, PREFIX(mpz_ptr) scratch);
__all__ void PREFIX(mpz_neg) (PREFIX(mpz_ptr) w, PREFIX(mpz_srcptr) u);
__all__ int PREFIX(mpn_cmp) (PREFIX(mp_srcptr) xp, PREFIX(mp_srcptr) yp, PREFIX(mp_size_t) size) __GMP_NOTHROW;

__all__ void PREFIX(mpz_add_strided)(PREFIX(mpz_ptr) w, PREFIX(mp_size_t) ws, PREFIX(mpz_srcptr) u, PREFIX(mp_size_t) us, PREFIX(mpz_srcptr) v, PREFIX(mp_size_t) vs);
__all__ void PREFIX(mpz_sub_strided)(PREFIX(mpz_ptr) w, PREFIX(mp_size_t) ws, PREFIX(mpz_srcptr) u, PREFIX(mp_size_t) us, PREFIX(mpz_srcptr) v, PREFIX(mp_size_t) vs);
__all__ int PREFIX(mpn_cmp_strided) (PREFIX(mp_srcptr) xp, PREFIX(mp_size_t) xs, PREFIX(mp_srcptr) yp, PREFIX(mp_size_t) ys, PREFIX(mp_size_t) size) __GMP_NOTHROW;
__all__ void PREFIX(mpz_mul_strided)(PREFIX(mpz_ptr) w, PREFIX(mp_size_t) ws, PREFIX(mpz_srcptr) u, PREFIX(mp_size_t) us, PREFIX(mpz_srcptr) v, PREFIX(mp_size_t) vs);
__all__ void PREFIX(mpz_mul_base_strided)(PREFIX(mpz_ptr) w, PREFIX(mp_size_t) ws, PREFIX(mpz_srcptr) u, PREFIX(mp_size_t) us, PREFIX(mpz_srcptr) v, PREFIX(mp_size_t) vs);
__all__ void PREFIX(mpz_addmul_strided)(PREFIX(mpz_ptr) w, PREFIX(mp_size_t) ws, PREFIX(mpz_srcptr) u, PREFIX(mp_size_t) us, PREFIX(mpz_srcptr) v, PREFIX(mp_size_t) vs, PREFIX(mpz_ptr) scratch, PREFIX(mp_size_t) ss);
__all__ void PREFIX(mpz_addmul_base_strided)(PREFIX(mpz_ptr) w, PREFIX(mp_size_t) ws, PREFIX(mpz_srcptr) u, PREFIX(mp_size_t) us, PREFIX(mpz_srcptr) v, PREFIX(mp_size_t) vs, PREFIX(mpz_ptr) scratch, PREFIX(mp_size_t) ss);

// Only asserts, do not call
__all__ PREFIX(mp_ptr) PREFIX(_mpz_realloc)(PREFIX(mpz_ptr) z, PREFIX(mp_size_t) n);

// vim: set sw=4 ts=4 noet:
