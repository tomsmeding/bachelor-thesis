#pragma once

#include <stdio.h>
#include "mat.h"
#include "job.h"


job_t matj_add_strided(unsigned int H, unsigned int W, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, struct job_dep_list_t gdeps);
job_t matj_sub_strided(unsigned int H, unsigned int W, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, struct job_dep_list_t gdeps);

job_t matj_mul_strided(unsigned int K, unsigned int M, unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, struct job_dep_list_t gdeps);
job_t matj_addmul_strided(unsigned int K, unsigned int M, unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, struct job_dep_list_t gdeps);
job_t matj_mul_strassen(unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps);
job_t matj_mul_winograd(unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps);
job_t matj_addmul_winograd(unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps);
job_t matj_mul_naive_winograd(unsigned int N, mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs, unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps);

job_t matj_mul_squarify(
		unsigned int K, unsigned int M, unsigned int N,
		mat_t C, mp_size_t Crowstride,
		mat_src_t A, mp_size_t Arowstride,
		mat_src_t B, mp_size_t Browstride,
		unsigned int basew, unsigned int baseh,
		unsigned int minblocksize, unsigned int minjobsize,
		job_t (*squarefunc)(
			unsigned int N,
			mat_t C, mp_size_t Crs, mat_src_t A, mp_size_t Ars, mat_src_t B, mp_size_t Brs,
			unsigned int minblocksize, unsigned int minjobsize, struct job_dep_list_t gdeps),
		struct job_dep_list_t gdeps);

// vim: set sw=4 ts=4 noet:
