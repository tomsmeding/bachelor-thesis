#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <assert.h>
#include "matj.h"
#include "util.h"
#include "topo.h"


static void usage(const char *argv0) {
	fprintf(stderr,
			"Usage: %s [-hgirp] [-n size] [-b bits] [-m method] [-B basesz] [-L loclvl]\n"
			"  -h         Show help\n"
			"  -g         Generate two random matrices and print them\n"
			"  -i         Read two matrices from input and multiply them\n"
			"  -r         Multiply two randomly generated matrices [default]\n"
			"\n"
			"For -g and -r:\n"
			"  -n K,M,N   Set the size for random matrix generation (KxM * MxN = KxN)\n"
			"  -b bits    Set the number of bits the generated random numbers should occupy\n"
			"  -s seed    Set the seed to use for random number generation\n"
			"\n"
			"For -i and -r:\n"
			"  -p         Print the multiplication result\n"
			"  -m method  Method to use for the multiplication (integer, see source)\n"
			"  -B basesz  Matrix size at which to switch to naive multiplication in a recursive algorithm\n"
			"  -J jobsz   Matrix size at which to switch to thread-local recursion in a job tree\n"
			"  -C ncores  Number of cores to use in concurrent job run\n",
			argv0);
}

// Returns floored result. Undefined if n == 0.
__attribute__((unused))
static mp_size_t ilog2(mp_size_t n) {
	return 8 * sizeof n - __builtin_clzll(n) - 1;
}


struct params {
	int method;
	int basesz, jobsz;
	int ncores;
};

static inline void perform_mul(unsigned int K, unsigned int M, unsigned int N, mat_t C, mat_src_t A, mat_src_t B, struct params params) {
	switch (params.method) {
		case 1:
			mat_mul(K, M, N, C, N, A, M, B, N);
			break;

		case 3:
		case 31:
		case 35: {
			if (params.basesz == -1) params.basesz = 32;
			if (params.jobsz == -1) params.jobsz = 128;
			if (params.ncores == -1) params.ncores = get_topology().nthreads;

			int64_t start = gettimestamp();
			switch (params.method) {
				case 3:
					matj_mul_squarify(K, M, N, C, N, A, M, B, N, params.basesz, params.basesz, params.basesz, params.jobsz, matj_mul_strassen, job_dep_list(0));
					break;
				case 31:
					matj_mul_squarify(K, M, N, C, N, A, M, B, N, params.basesz, params.basesz, params.basesz, params.jobsz, matj_mul_naive_winograd, job_dep_list(0));
					break;
				case 35:
					matj_mul_squarify(K, M, N, C, N, A, M, B, N, params.basesz, params.basesz, params.basesz, params.jobsz, matj_mul_winograd, job_dep_list(0));
					break;
			}
			int64_t mid = gettimestamp();
			fprintf(stderr, "Preparation took %g seconds\n", (double)(mid - start) / 1000000);
			// job_run_linear();
			job_run_concur(params.ncores);
			fprintf(stderr, "Execution took %g seconds\n", (double)(gettimestamp() - mid) / 1000000);
			break;
		}

		default:
			fprintf(stderr, "Unknown multiplication method %d\n", params.method);
			exit(1);
	}
}

static unsigned long intarg(const char *arg, const char *opt) {
	char *endp;
	unsigned long value = strtoul(arg, &endp, 10);
	if (*arg == '\0' || *endp != '\0') {
		fprintf(stderr, "Invalid number for %s\n", opt);
		return 1;
	}
	return value;
}


int main(int argc, char **argv) {
	enum mode {
		MODE_GENERATE,
		MODE_MULT_RANDOM,
		MODE_MULT_INPUT,
	};

	enum mode mode = MODE_MULT_RANDOM;
	int K = -1, M = -1, N = -1, nbits = -1;
	unsigned long seed = (unsigned long)-1;
	bool print_result = false;
	struct params params = {-1, -1, -1, -1};

	{
		int opt;
		while ((opt = getopt(argc, argv, "hgirpn:b:s:m:B:J:C:")) != -1) {
			switch (opt) {
				case 'h':
					usage(argv[0]);
					return 0;

				case 'g':
					mode = MODE_GENERATE;
					break;

				case 'i':
					mode = MODE_MULT_INPUT;
					break;

				case 'r':
					mode = MODE_MULT_RANDOM;
					break;

				case 'p':
					print_result = true;
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

				case 's':
					seed = intarg(optarg, "-s");
					break;

				case 'm':
					params.method = intarg(optarg, "-m");
					break;

				case 'B':
					params.basesz = intarg(optarg, "-B");
					break;

				case 'J':
					params.jobsz = intarg(optarg, "-J");
					break;

				case 'C':
					params.ncores = intarg(optarg, "-C");
					break;

				default:
					usage(argv[0]);
					return 1;
			}
		}
	}

	if (mode == MODE_MULT_INPUT) {
		if (K != -1 || nbits != -1) {
			fprintf(stderr, "Cannot specify -n or -b in combination with -i\n");
			return 1;
		}
	} else {
		if (K <= -1) {
			fprintf(stderr, "No matrix size specified!\n");
			return 1;
		}
		if (nbits <= -1) {
			fprintf(stderr, "No bit count specified!\n");
			return 1;
		}
	}

	if (params.method == -1 && mode != MODE_GENERATE) {
		fprintf(stderr, "Defaulting to method 31\n");
		params.method = 31;
	}

	if (seed != (unsigned long)-1) {
		fprintf(stderr, "Seeding with value %lu\n", seed);
		mat_seed_generator(seed);
	}


	mat_t A = NULL, B = NULL, C = NULL;

	int64_t timediff = -1;

	switch (mode) {
		case MODE_MULT_INPUT: {
			scanf("%u %u %u", &K, &M, &N);

			A = mat_init(M, K);
			B = mat_init(N, M);
			C = mat_init(N, K);

			mat_read(M, K, stdin, A);
			mat_read(N, M, stdin, B);

			int64_t start = gettimestamp();
			perform_mul(K, M, N, C, A, B, params);
			timediff = gettimestamp() - start;
			break;
		}

		case MODE_MULT_RANDOM: {
			A = mat_init(M, K);
			B = mat_init(N, M);
			C = mat_init(N, K);

			mat_random(M, K, A, nbits);
			mat_random(N, M, B, nbits);

			int64_t start = gettimestamp();
			perform_mul(K, M, N, C, A, B, params);
			timediff = gettimestamp() - start;
			break;
		}

		case MODE_GENERATE: {
			printf("%u %u %u\n", K, M, N);

			mpz_t z;
			mpz_init(z);
			for (int y = 0; y < K; y++) {
				for (int x = 0; x < M; x++) {
					mat_random_mpz(z, nbits);
					if (x != 0) printf(" ");
					mpz_out_str(stdout, 16, z);
				}
				printf("\n");
			}
			printf("\n");

			for (int y = 0; y < M; y++) {
				for (int x = 0; x < N; x++) {
					mat_random_mpz(z, nbits);
					if (x != 0) printf(" ");
					mpz_out_str(stdout, 16, z);
				}
				printf("\n");
			}
			printf("\n");

			mpz_clear(z);
			break;
		}
	}

	if (timediff != -1) {
		fprintf(stderr, "Multiplication took %g seconds\n", (double)timediff / 1000000);
	}

	if (print_result) {
		assert(C != NULL);
		mat_write(N, K, stdout, C, N);
	}

	if (A != NULL) {mat_free(K, M, A); A = NULL;}
	if (B != NULL) {mat_free(M, N, B); B = NULL;}
	if (C != NULL) {mat_free(K, N, C); C = NULL;}
}

// vim: set sw=4 ts=4 noet:
