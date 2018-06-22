#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include "gmp-es.h"
#include "gmp-es-mpz.h"
#include "cusuc.h"
#include "job.h"
#include "util.h"
#include "mat-params.h"
#include "mat-defines.h"
#include "mat-gpu.h"
#include "mat-gpu-launchers.h"
#include "mat-pregpu.h"
#include "mat-core.h"


// Rowstride values count struct __mpz_t's; limbstride values count mp_limb_t's.


template <typename T>
static std::ostream& operator<<(std::ostream &os, const std::vector<T> &v) {
	os << '{';
	switch (v.size()) {
		case 0: return os << '}';
		case 1: return os << v[0] << '}';
		default:
			bool first = true;
			for (const T &t : v) {
				if (first) first = false;
				else os << ", ";
				os << t;
			}
			return os << '}';
	}
}

__attribute__((unused))
static void xxd(void *ptr_, size_t size) {
	uint8_t *ptr = (uint8_t*)ptr_;
	for (size_t i = 0; i < size; i += 16) {
		printf("%08zx:", i);
		bool zeromark = false;
		for (size_t j = 0; j < 16; j++) {
			if (j % 2 == 0) printf(" ");
			if (j % 4 == 0 && i + j + 4 <= size) {
				if (*(uint32_t*)&ptr[i + j] == 0) {
					if (!zeromark) printf("\x1B[36m");
					zeromark = true;
				} else {
					if (zeromark) printf("\x1B[0m");
					zeromark = false;
				}
			}
			printf("%02x", ptr[i + j]);
			if (i + j + 1 >= size) break;
		}
		if (zeromark) printf("\x1B[0m");
		printf("\n");
	}
}

__attribute__((unused))
static void launcher_mul_cpu(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	for (gpu_mp_size_t y = 0; y < K; y++) {
		for (gpu_mp_size_t x = 0; x < N; x++) {
			mat_mul_1(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, x, y);
		}
	}
}

__attribute__((unused))
static void launcher_mul_strided_cpu(
			gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	for (gpu_mp_size_t y = 0; y < K; y++) {
		for (gpu_mp_size_t x = 0; x < N; x++) {
			mat_mul_1_strided(K, M, N, C, Crowstride, A, Arowstride, B, Browstride, x, y);
		}
	}
}

__attribute__((unused))
static void launcher_add_cpu(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	for (gpu_mp_size_t y = 0; y < H; y++) {
		for (gpu_mp_size_t x = 0; x < W; x++) {
			gpu_mpz_add(&C[Crowstride * y + x], &A[Arowstride * y + x], &B[Browstride * y + x]);
		}
	}
}

__attribute__((unused))
static void launcher_sub_cpu(
			gpu_mp_size_t W, gpu_mp_size_t H,
			struct gpu___mpz_t *C, gpu_mp_size_t Crowstride,
			const struct gpu___mpz_t *A, gpu_mp_size_t Arowstride,
			const struct gpu___mpz_t *B, gpu_mp_size_t Browstride) {

	for (gpu_mp_size_t y = 0; y < H; y++) {
		for (gpu_mp_size_t x = 0; x < W; x++) {
			gpu_mpz_sub(&C[Crowstride * y + x], &A[Arowstride * y + x], &B[Browstride * y + x]);
		}
	}
}

static void cpu_mult(gpu_mp_size_t K, gpu_mp_size_t M, gpu_mp_size_t N, struct gpu___mpz_t *C, const struct gpu___mpz_t *A, const struct gpu___mpz_t *B) {
	for (gpu_mp_size_t y = 0; y < K; y++) {
		for (gpu_mp_size_t x = 0; x < N; x++) {
			mat_mul_1(K, M, N, C, N, A, M, B, N, x, y);
		}
	}
}

// Assumes p_ points to ngroup groups of npart buffers, each of partsz bytes. Within
// each group, the npart buffers are interleaved in blocks of atomsz bytes.
template <size_t atomsz>
static void transform_interleave(void *p_, size_t ngroup, size_t npart, size_t partsz) {
	assert(partsz % atomsz == 0);
	size_t natom = partsz / atomsz;

	uint8_t *tempb = (uint8_t*)malloc(npart * partsz);

	uint8_t *p = (uint8_t*)p_;
	for (size_t i = 0; i < ngroup; i++) {
		for (size_t part = 0; part < npart; part++) {
			for (size_t atom = 0; atom < natom; atom++) {
				memcpy(tempb + atomsz * (npart * atom + part), p + atomsz * (natom * part + atom), atomsz);
			}
		}
		memcpy(p, tempb, npart * partsz);
		p += npart * partsz;
	}

	free(tempb);
}

static int intarg(const char *arg, const char *opt) {
	char *endp;
	int value = strtoul(arg, &endp, 10);
	if (*arg == '\0' || *endp != '\0') {
		fprintf(stderr, "Invalid number for %s\n", opt);
		return 1;
	}
	return value;
}

__attribute__((unused))
static void test_interleave(void) {
	const int nbits = 64;
	const int N = INTERLEAVE_BLOCK_SIZE;

	gpu_gmp_randstate_t rs;
	gpu_gmp_randinit_default(rs);

	for (int i = 0; i < 10000; i++) {
		gpu_mpz_t A[N], B[N], C[N];
		for (int i = 0; i < N; i++) {
			gpu_mpz_init(A[i]);
			gpu_mpz_init(B[i]);
			gpu_mpz_init(C[i]);
		}

		gpu_mpz_urandomb(A[0], rs, nbits);
		gpu_mpz_urandomb(B[0], rs, nbits);
		if (gpu_gmp_urandomb_ui(rs, 1) == 0) gpu_mpz_neg(A[0], A[0]);
		if (gpu_gmp_urandomb_ui(rs, 1) == 0) gpu_mpz_neg(B[0], B[0]);
		if (gpu_gmp_urandomb_ui(rs, 1) == 0) gpu_mpz_neg(C[0], C[0]);

		gpu_mpz_t scratch;

		printf("0x");
		gpu_mpz_out_file(stdout, C[0]);
		printf(" + 0x");
		gpu_mpz_out_file(stdout, A[0]);
		printf(" * 0x");
		gpu_mpz_out_file(stdout, B[0]);
		printf(" == 0x");

		transform_interleave<sizeof(gpu_mp_limb_t)>
			(A, N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));
		transform_interleave<sizeof(gpu_mp_limb_t)>
			(B, N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));
		transform_interleave<sizeof(gpu_mp_limb_t)>
			(C, N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));

		gpu_mpz_addmul_strided(&C[0][0], INTERLEAVE_BLOCK_SIZE, &A[0][0], INTERLEAVE_BLOCK_SIZE, &B[0][0], INTERLEAVE_BLOCK_SIZE, scratch, 1);

		transform_interleave<sizeof(gpu_mp_limb_t)>
			(C, N / INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t) / sizeof(gpu_mp_limb_t), INTERLEAVE_BLOCK_SIZE * sizeof(gpu_mp_limb_t));

		gpu_mpz_out_file(stdout, C[0]);
		printf("\n");
	}
}

#define TIMER_INIT int64_t __timer_timestamp = gettimestamp()
#define TIMER_POINT(str_) do { int64_t t = gettimestamp(); fprintf(stderr, "! %s (%g)\n", (str_), (double)(t - __timer_timestamp) / 1000000); __timer_timestamp = t; } while (0)

int main(int argc, char **argv) {
	// test_interleave();
	// return 0;

	srandom(time(NULL));

	bool read_matrices = false, print_result = false;
	gpu_mp_size_t K = -1, M = -1, N = -1;
	gpu_mp_size_t nbits = -1;
	int method = 1;

	TIMER_INIT;

	{
		int opt;
		while ((opt = getopt(argc, argv, "hin:b:m:p")) != -1) {
			switch (opt) {
				case 'h':
					fprintf(stderr,
							"Usage: %s [-hip] [-n size] [-b nbits] [-m method]\n"
							"  -h         Show help\n"
							"  -i         Read matrices from input instead of generating randomly\n"
							"  -n K,M,N   Set the size for random matrix generation (KxM * MxN = KxN)\n"
							"  -b nbits   Number of bits for generated numbers\n"
							"  -m method  Use specified method for multiplication, see source\n"
							"  -p         Print result after multiplication\n",
							argv[0]);
					return 0;

				case 'i':
					read_matrices = true;
					break;

				case 'n': {
					int len = strlen(optarg);
					if (len >= 64) {
						fprintf(stderr, "Too long argument for -n\n");
						return 1;
					}
					char buf[64];
					strcpy(buf, optarg);
					char *p = strchr(buf, ','), *q;
					if (p == NULL || (q = strchr(p + 1, ',')) == NULL || strchr(q + 1, ',') != NULL) {
						fprintf(stderr, "Invalid number of commas in argument for -n\n");
						return 1;
					}
					*p = *q = '\0';
					K = intarg(buf, "-n");
					M = intarg(p + 1, "-n");
					N = intarg(q + 1, "-n");
					break;
				}

				case 'b':
					nbits = intarg(optarg, "-b");
					break;

				case 'm':
					method = intarg(optarg, "-m");
					break;

				case 'p':
					print_result = true;
					break;

				default:
					fprintf(stderr, "Unknown flag, use -h to get info\n");
					return 1;
			}
		}
	}

	if (read_matrices) {
		if (K != -1) {
			fprintf(stderr, "Cannot specify matrix size if reading matrices from input\n");
			return 1;
		}
		if (nbits != -1) {
			fprintf(stderr, "Cannot specify nbits if reading matrices from input\n");
			return 1;
		}
		scanf("%u %u %u", &K, &M, &N);
	} else {
		if (nbits == -1) {
			fprintf(stderr, "No bit count specified!\n");
			return 1;
		}

		assert(2 * nbits + log2((double)max(max(K, M), N)) + 1 <= NUM_SIZE);
	}

	if (K == -1) {
		fprintf(stderr, "No matrix size specified!\n");
		return 1;
	}

	if (method == -1) {
		fprintf(stderr, "Defaulting to method 3\n");
		method = 3;
	}

	CUSUC(cudaSetDevice(0));

	struct gpu___mpz_t *A, *B, *C;
	CUSUC(cudaHostAlloc(&A, M * K * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));
	CUSUC(cudaHostAlloc(&B, N * M * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));
	CUSUC(cudaHostAlloc(&C, N * K * sizeof(struct gpu___mpz_t), cudaHostAllocDefault));

	TIMER_POINT("Allocate");

	memset(A, 0, M * K * sizeof(struct gpu___mpz_t));
	memset(B, 0, N * M * sizeof(struct gpu___mpz_t));
	memset(C, 0, N * K * sizeof(struct gpu___mpz_t));

	TIMER_POINT("Memset");

	if (read_matrices) {
		for (gpu_mp_size_t i = 0; i < M * K; i++) gpu_mpz_inp_file(&A[i], stdin);
		for (gpu_mp_size_t i = 0; i < N * M; i++) gpu_mpz_inp_file(&B[i], stdin);
	} else {
		gpu_gmp_randstate_t rs;
		gpu_gmp_randinit_default(rs);
		gpu_gmp_randinit_seed(rs, time(NULL));

		for (gpu_mp_size_t i = 0; i < M * K; i++) {
			gpu_mpz_urandomb(&A[i], rs, nbits);
			if (gpu_gmp_urandomb_ui(rs, 1) == 0) gpu_mpz_neg(&A[i], &A[i]);
		}
		for (gpu_mp_size_t i = 0; i < N * M; i++) {
			gpu_mpz_urandomb(&B[i], rs, nbits);
			if (gpu_gmp_urandomb_ui(rs, 1) == 0) gpu_mpz_neg(&A[i], &A[i]);
		}
	}

	TIMER_POINT("Read/Generate");

	fprintf(stderr, "Using method %d\n", method);
	int64_t start = gettimestamp();
	switch (method) {
		case 1:
			cpu_mult(K, M, N, C, A, B);
			break;

		case 2: {
			CUSUC(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

			struct gpu___mpz_t *devA, *devB, *devC;
			CUSUC(cudaMalloc(&devA, M * K * sizeof(struct gpu___mpz_t)));
			CUSUC(cudaMalloc(&devB, N * M * sizeof(struct gpu___mpz_t)));
			CUSUC(cudaMalloc(&devC, N * K * sizeof(struct gpu___mpz_t)));
			CUSUC(cudaMemcpy(devA, A, M * K * sizeof(struct gpu___mpz_t), cudaMemcpyHostToDevice));
			CUSUC(cudaMemcpy(devB, B, N * M * sizeof(struct gpu___mpz_t), cudaMemcpyHostToDevice));

			launcher_mul<<<dim3(N / GPU_BLK_W, K / GPU_BLK_H), dim3(GPU_BLK_W, GPU_BLK_H)>>>(
					K, M, N, devC, N, devA, M, devB, N);
			CUSUC(cudaDeviceSynchronize());

			CUSUC(cudaMemcpy(C, devC, N * K * sizeof(struct gpu___mpz_t), cudaMemcpyDeviceToHost));

			CUSUC(cudaFree(devA));
			CUSUC(cudaFree(devB));
			CUSUC(cudaFree(devC));
			break;
		}

		case 3: {
			CUSUC(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

			/*assert(K == M && M == N);
			strassen_pregpu(K, C, N, A, M, B, N);*/
			squarify_mul_pregpu(K, M, N, C, N, A, M, B, N, strassen_pregpu);
			break;
		}

		case 37: {
			assert(K == M && M == N);

			int64_t t1 = gettimestamp(), t2;

			CUSUC(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

			t2 = gettimestamp(); fprintf(stderr, "$prepare: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			assert(N % INTERLEAVE_BLOCK_SIZE == 0);
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(C, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(A, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(B, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));

			t2 = gettimestamp(); fprintf(stderr, "$interleave: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			strassen_strided_pregpu(N, C, N, A, N, B, N);

			t2 = gettimestamp(); fprintf(stderr, "$multiply: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			transform_interleave<sizeof(gpu_mp_limb_t)>
				(C, N * N / INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t) / sizeof(gpu_mp_limb_t), INTERLEAVE_BLOCK_SIZE * sizeof(gpu_mp_limb_t));

			t2 = gettimestamp(); fprintf(stderr, "$deinterleave: %g\n", (t2 - t1) / 1000000.0); t1 = t2;
			break;
		}

		case 35: {
			CUSUC(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

			squarify_mul_pregpu(K, M, N, C, N, A, M, B, N, winograd_pregpu);
			break;
		}

		case 357: {
			assert(K == M && M == N);

			int64_t t1 = gettimestamp(), t2;

			CUSUC(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

			t2 = gettimestamp(); fprintf(stderr, "$prepare: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			assert(N % INTERLEAVE_BLOCK_SIZE == 0);
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(C, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(A, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(B, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));

			t2 = gettimestamp(); fprintf(stderr, "$interleave: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			winograd_strided_pregpu(N, C, N, A, N, B, N);

			t2 = gettimestamp(); fprintf(stderr, "$multiply: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			transform_interleave<sizeof(gpu_mp_limb_t)>
				(C, N * N / INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t) / sizeof(gpu_mp_limb_t), INTERLEAVE_BLOCK_SIZE * sizeof(gpu_mp_limb_t));

			t2 = gettimestamp(); fprintf(stderr, "$deinterleave: %g\n", (t2 - t1) / 1000000.0); t1 = t2;
			break;
		}

		case 4: {
			CUSUC(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

			squarify_mul_pregpu(K, M, N, C, N, A, M, B, N, naive_recursive_pregpu);
			break;
		}

		case 47: {
			assert(K == M && M == N);

			int64_t t1 = gettimestamp(), t2;

			CUSUC(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

			t2 = gettimestamp(); fprintf(stderr, "$prepare: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			assert(N % INTERLEAVE_BLOCK_SIZE == 0);
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(C, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(A, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));
			transform_interleave<sizeof(gpu_mp_limb_t)>
				(B, N * N / INTERLEAVE_BLOCK_SIZE, INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t));

			t2 = gettimestamp(); fprintf(stderr, "$interleave: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			naive_recursive_strided_pregpu(N, C, N, A, N, B, N);
			// launcher_mul_strided_cpu(N, C, N, A, N, B, N);

			t2 = gettimestamp(); fprintf(stderr, "$multiply: %g\n", (t2 - t1) / 1000000.0); t1 = t2;

			transform_interleave<sizeof(gpu_mp_limb_t)>
				(C, N * N / INTERLEAVE_BLOCK_SIZE, sizeof(struct gpu___mpz_t) / sizeof(gpu_mp_limb_t), INTERLEAVE_BLOCK_SIZE * sizeof(gpu_mp_limb_t));

			t2 = gettimestamp(); fprintf(stderr, "$deinterleave: %g\n", (t2 - t1) / 1000000.0); t1 = t2;
			break;
		}

		default:
			fprintf(stderr, "Invalid method %d!\n", method);
			return 1;
	}
	int64_t diff = gettimestamp() - start;
	fprintf(stderr, "Time taken: %g seconds\n", (double)diff / 1000000);

	TIMER_POINT("Compute");

	CUSUC(cudaFreeHost(A));
	CUSUC(cudaFreeHost(B));

	TIMER_POINT("Free AB");

	if (print_result) {
		for (gpu_mp_size_t y = 0; y < K; y++) {
			for (gpu_mp_size_t x = 0; x < N; x++) {
				if (x != 0) printf(" ");
				gpu_mpz_out_file(stdout, &C[N * y + x]);
			}
			printf("\n");
		}
		printf("\n");

		TIMER_POINT("Printing");
	}

	CUSUC(cudaFreeHost(C));

	TIMER_POINT("Free C");

	CUSUC(cudaDeviceReset());

	TIMER_POINT("Reset");
}

// vim: set sw=4 ts=4 noet:
