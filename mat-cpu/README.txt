Large-integer matrix multiplication on the CPU.

gmp.h is the header from the exact version of the source tree that
libgmp_asm_tuned.a was compiled from, which is GMP 6.1.2. It's compiled for Arch
Linux somewhere around May 2018.

The -flto flag is required for remove_factors(_, 2) to be inlined.
