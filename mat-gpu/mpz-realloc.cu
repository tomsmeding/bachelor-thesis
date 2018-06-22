#include <assert.h>
#include "gmp-es-mpz.h"
#include "gmp-es-internal-convert.h"

__all__ mp_ptr
PREFIX(_mpz_realloc)(mpz_ptr z, mp_size_t n)
{
  (void)z; (void)n;
  assert(false);
  return NULL;
}

// vim: set sw=2 ts=8 noet:
