#pragma once

#include <assert.h>
#include "gmp-es-mpz.h"


__all__ inline void mat_addmul_1(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride,
			gpu_mp_size_t x, gpu_mp_size_t y) {

	(void)K; (void)N;

	gpu_mp_size_t Cidx = Crowstride * y + x;
	gpu_mp_size_t Aidx = Arowstride * y + 0;
	gpu_mp_size_t Bidx = Browstride * 0 + x;

	gpu_mpz_t scratch;

	gpu_mpz_addmul(&C[Cidx], &A[Aidx], &B[Bidx], scratch);

	for (gpu_mp_size_t k = 1; k < M; k++) {
		Aidx++;
		Bidx += Browstride;
		gpu_mpz_addmul(&C[Cidx], &A[Aidx], &B[Bidx], scratch);
	}
}

__all__ inline void mat_mul_1(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride,
			gpu_mp_size_t x, gpu_mp_size_t y) {

	(void)K; (void)N;

	gpu_mp_size_t Cidx = Crowstride * y + x;
	gpu_mp_size_t Aidx = Arowstride * y + 0;
	gpu_mp_size_t Bidx = Browstride * 0 + x;

	gpu_mpz_t scratch;

	gpu_mpz_mul(&C[Cidx], &A[Aidx], &B[Bidx]);

	for (gpu_mp_size_t k = 1; k < M; k++) {
		Aidx++;
		Bidx += Browstride;
		gpu_mpz_addmul(&C[Cidx], &A[Aidx], &B[Bidx], scratch);
	}
}

__all__ inline void mat_mul_1_strided(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride,
			gpu_mp_size_t x, gpu_mp_size_t y) {

	(void)K; (void)N;

	const gpu_mp_size_t Alimbstride = INTERLEAVE_BLOCK_SIZE;
	const gpu_mp_size_t Blimbstride = INTERLEAVE_BLOCK_SIZE;
	const gpu_mp_size_t Climbstride = INTERLEAVE_BLOCK_SIZE;

	// limbstride is, besides the limb array stride, also the amount of interleaved gpu_mpz_t's in a
	// chunk. rowstride is the amount of gpu_mpz_t's in a row, so the below inequalities should hold
	// for a chunk to not span multiple rows.
	assert(Crowstride % Climbstride == 0);
	assert(Arowstride % Alimbstride == 0);
	assert(Browstride % Blimbstride == 0);

	gpu_mp_limb_t *Cp = (gpu_mp_limb_t*)&C[Crowstride * y + x / Climbstride * Climbstride] + x % Climbstride;
	const gpu_mp_limb_t *Ap = (gpu_mp_limb_t*)&A[Arowstride * y + 0 / Alimbstride * Alimbstride] + 0 % Alimbstride;
	const gpu_mp_limb_t *Bp = (gpu_mp_limb_t*)&B[Browstride * 0 + x / Blimbstride * Blimbstride] + x % Blimbstride;

	struct gpu___mpz_t *Czp = (struct gpu___mpz_t*)Cp;
	const struct gpu___mpz_t *Azp = (struct gpu___mpz_t*)Ap;
	const struct gpu___mpz_t *Bzp = (struct gpu___mpz_t*)Bp;

	const struct gpu___mpz_t *Azpstart = Azp;  // start of current interleave block

	gpu_mpz_t scratch;

	gpu_mpz_mul_strided(Czp, Climbstride, Azp, Alimbstride, Bzp, Blimbstride);

	for (gpu_mp_size_t k = 1; k < M; k++) {
		if (k % Alimbstride == 0) Azp = (Azpstart += Alimbstride);
		else Azp = (struct gpu___mpz_t*)((gpu_mp_limb_t*)Azp + 1);
		Bzp += Browstride;
		gpu_mpz_addmul_strided(Czp, Climbstride, Azp, Alimbstride, Bzp, Blimbstride, scratch, 1);
	}
}

__all__ inline void mat_addmul_1_strided(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride,
			gpu_mp_size_t x, gpu_mp_size_t y) {

	(void)K; (void)N;

	const gpu_mp_size_t Alimbstride = INTERLEAVE_BLOCK_SIZE;
	const gpu_mp_size_t Blimbstride = INTERLEAVE_BLOCK_SIZE;
	const gpu_mp_size_t Climbstride = INTERLEAVE_BLOCK_SIZE;

	// limbstride is, besides the limb array stride, also the amount of interleaved gpu_mpz_t's in a
	// chunk. rowstride is the amount of gpu_mpz_t's in a row, so the below inequalities should hold
	// for a chunk to not span multiple rows.
	assert(Climbstride <= Crowstride);
	assert(Alimbstride <= Arowstride);
	assert(Blimbstride <= Browstride);

	gpu_mp_limb_t *Cp = (gpu_mp_limb_t*)&C[Crowstride * y + x / Climbstride * Climbstride] + x % Climbstride;
	const gpu_mp_limb_t *Ap = (gpu_mp_limb_t*)&A[Arowstride * y + 0 / Alimbstride * Alimbstride] + 0 % Alimbstride;
	const gpu_mp_limb_t *Bp = (gpu_mp_limb_t*)&B[Browstride * 0 + x / Blimbstride * Blimbstride] + x % Blimbstride;

	struct gpu___mpz_t *Czp = (struct gpu___mpz_t*)Cp;
	const struct gpu___mpz_t *Azp = (struct gpu___mpz_t*)Ap;
	const struct gpu___mpz_t *Bzp = (struct gpu___mpz_t*)Bp;

	const struct gpu___mpz_t *Azpstart = Azp;  // start of current interleave block

	gpu_mpz_t scratch;

	gpu_mpz_addmul_strided(Czp, Climbstride, Azp, Alimbstride, Bzp, Blimbstride, scratch, 1);

	for (gpu_mp_size_t k = 1; k < M; k++) {
		if (k % Alimbstride == 0) Azp = (Azpstart += Alimbstride);
		else Azp = (struct gpu___mpz_t*)((gpu_mp_limb_t*)Azp + 1);
		Bzp += Browstride;
		gpu_mpz_addmul_strided(Czp, Climbstride, Azp, Alimbstride, Bzp, Blimbstride, scratch, 1);
	}
}

// vim: set sw=4 ts=4 noet:
