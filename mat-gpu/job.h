#pragma once

#include <vector>
#include <functional>


typedef int job_t;

struct job_data_t {
	cudaStream_t stream;
	std::vector<void*> chunks;
};

void job_set_chunk_size(size_t bytes);

// mem_req: required bytes of GPU memory
job_t job_submit(
		size_t num_chunks,
		const std::vector<job_t> &deps,
		const std::function<void(struct job_data_t)> &pre_callback);

void job_add_dep(job_t id, job_t dep);
void job_add_dep(job_t id, std::vector<job_t> deps);
void job_add_dep(std::vector<job_t> ids, std::vector<job_t> deps);

void job_run_linear(void);
void job_run_concur(void);


template <typename T>
void append(std::vector<T> &d, const std::vector<T> &s) {
	d.insert(d.end(), s.begin(), s.end());
}

// vim: set sw=4 ts=4 noet:
