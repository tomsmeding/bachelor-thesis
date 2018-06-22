#include <assert.h>
#include <unistd.h>
#include <unordered_map>
#include <algorithm>
#include "job.h"
#include "cusuc.h"


struct job_info_t {
	size_t num_chunks;
	std::function<void(struct job_data_t)> pre_callback;
	std::vector<job_t> deps, revdeps;
};

static std::unordered_map<job_t, struct job_info_t> jobmap;

static std::vector<job_t> source_jobs;

static size_t chunk_size = 0;


static job_t gen_job_id(void) {
	static job_t id = 0;
	return id++;
}


void job_set_chunk_size(size_t bytes) {
	chunk_size = bytes;
}

job_t job_submit(
		size_t num_chunks,
		const std::vector<job_t> &deps,
		const std::function<void(struct job_data_t)> &pre_callback) {

	job_t id = gen_job_id();

	struct job_info_t info;
	info.num_chunks = num_chunks;
	info.pre_callback = pre_callback;

	for (job_t dep : deps) {
		info.deps.push_back(dep);
		jobmap.find(dep)->second.revdeps.push_back(id);
	}

	if (info.deps.size() == 0) {
		source_jobs.push_back(id);
	}

	jobmap.emplace(id, std::move(info));

	return id;
}

void job_add_dep(job_t id, job_t dep) {
	struct job_info_t &info = jobmap.find(id)->second;
	if (info.deps.size() == 0) {
		source_jobs.erase(std::find(source_jobs.begin(), source_jobs.end(), id));
	}
	info.deps.push_back(dep);
	jobmap.find(dep)->second.revdeps.push_back(id);
}

void job_add_dep(job_t id, std::vector<job_t> deps) {
	struct job_info_t &info = jobmap.find(id)->second;
	if (info.deps.size() == 0) {
		source_jobs.erase(std::find(source_jobs.begin(), source_jobs.end(), id));
	}
	info.deps.insert(info.deps.end(), deps.begin(), deps.end());
	for (job_t d : deps) {
		jobmap.find(d)->second.revdeps.push_back(id);
	}
}

void job_add_dep(std::vector<job_t> ids, std::vector<job_t> deps) {
	for (job_t id : ids) {
		job_add_dep(id, deps);
	}
}

void job_run_linear(void) {
	fprintf(stderr, "-- Linear job run, %zu jobs\n", jobmap.size());

	assert(chunk_size > 0);

	std::vector<void*> devChunks;
	while (true) {
		void *ptr;
		cudaError_t err = cudaMalloc(&ptr, chunk_size);
		if (err == cudaErrorMemoryAllocation) break;
		CUSUC(err);
		devChunks.push_back(ptr);

		if (devChunks.size() >= 3) break;  // TODO fix
	}

	fprintf(stderr, "-- Allocated %zu chunks of %zu bytes each\n", devChunks.size(), chunk_size);

	while (source_jobs.size() > 0) {
		// fprintf(stderr, "-- source_jobs = {");
		// for (job_t j : source_jobs) fprintf(stderr, "%d,", j);
		// fprintf(stderr, "}\n");

		job_t id = source_jobs[0];
		source_jobs.erase(source_jobs.begin());

		fprintf(stderr, "-- Running job %d\n", id);

		struct job_info_t &info = jobmap.find(id)->second;
		assert(info.num_chunks <= devChunks.size());

		struct job_data_t data;
		data.stream = nullptr;
		data.chunks = std::vector<void*>(devChunks.begin(), devChunks.begin() + info.num_chunks);

		info.pre_callback(data);
		CUSUC(cudaStreamSynchronize(data.stream));

		// fprintf(stderr, "-- revdeps = {");
		// for (job_t j : info.revdeps) fprintf(stderr, "%d,", j);
		// fprintf(stderr, "}\n");

		std::vector<job_t> new_sources;

		for (job_t rd : info.revdeps) {
			struct job_info_t &info2 = jobmap.find(rd)->second;

			// fprintf(stderr, "-- deps(%d) = {", rd);
			// for (job_t j : info2.deps) fprintf(stderr, "%d,", j);
			// fprintf(stderr, "}\n");

			auto it = std::find(info2.deps.begin(), info2.deps.end(), id);
			info2.deps.erase(it);
			if (info2.deps.size() == 0) {
				new_sources.push_back(rd);
			}
		}

		source_jobs.insert(source_jobs.begin(), new_sources.begin(), new_sources.end());

		jobmap.erase(jobmap.find(id));
	}

	for (void *ptr : devChunks) {
		CUSUC(cudaFree(ptr));
	}

	if (jobmap.size() == 0) {
		fprintf(stderr, "-- All jobs ran\n");
	} else {
		fprintf(stderr, "-- WARNING: NOT ALL JOBS RAN!\n");
	}
}

struct stream_info_t {
	job_t job;
	cudaStream_t stream;
	std::vector<size_t> chunks;  // indices in devChunks
	struct job_data_t data;
};

__attribute__((unused))
static void print_running_bars(const std::vector<struct stream_info_t> &vec) {
	std::vector<bool> mark;
	for (const struct stream_info_t &info : vec) {
		if (mark.size() <= info.job) {
			mark.resize(info.job + 1);
		}
		mark[info.job] = true;
	}
	fprintf(stderr, "BARS:");
	for (size_t i = 0; i < mark.size(); i++) {
		const bool b = mark[i];
		// if (b) fprintf(stderr, "#");
		// else fprintf(stderr, " ");
		if (b) fprintf(stderr, " %zu", i);
	}
	fprintf(stderr, "\n");
}

void job_run_concur(void) {
	fprintf(stderr, "-- Concurrent job run, %zu jobs\n", jobmap.size());

	assert(chunk_size > 0);

	std::vector<void*> devChunks;
	while (true) {
		void *ptr;
		cudaError_t err = cudaMalloc(&ptr, chunk_size);
		if (err == cudaErrorMemoryAllocation) break;
		CUSUC(err);
		devChunks.push_back(ptr);

		if (devChunks.size() >= 9) break;  // TODO fix
	}

	std::vector<bool> chunk_taken(devChunks.size());
	size_t num_chunks_available = devChunks.size();

	std::vector<struct stream_info_t> running_streams;

	fprintf(stderr, "-- Allocated %zu chunks of %zu bytes each\n", devChunks.size(), chunk_size);

#define MAX_CONCURRENCY 4

#define TOTAL_TO_RUN 999999
	size_t total_started = 0;

	while (source_jobs.size() > 0 || running_streams.size() > 0) {
		// fprintf(stderr, "-- source_jobs =");
		// for (job_t j : source_jobs) fprintf(stderr, " %d(%dc)", j, jobmap.find(j)->second.num_chunks);
		// fprintf(stderr, "\n");
		// fprintf(stderr, "-- %zu chunks available\n", num_chunks_available);

		// print_running_bars(running_streams);

		if (total_started >= TOTAL_TO_RUN && running_streams.size() == 0) break;

		job_t id;
		size_t index;
		for (index = 0; index < source_jobs.size(); index++) {
			id = source_jobs[index];
			if (jobmap.find(id)->second.num_chunks <= num_chunks_available) break;
		}

		if (index == source_jobs.size() || running_streams.size() >= MAX_CONCURRENCY || total_started >= TOTAL_TO_RUN) {
			// fprintf(stderr, "-- No new task fits\n");
			bool was_ready = false;

			for (size_t i = 0; i < running_streams.size(); i++) {
				struct stream_info_t &s = running_streams[i];

				cudaError_t err = cudaStreamQuery(s.stream);

				if (err == cudaSuccess) {
					fprintf(stderr, "-- Job %d has finished stream\n", s.job);

					struct job_info_t info = jobmap.find(s.job)->second;

					for (size_t chidx : s.chunks) {
						chunk_taken[chidx] = false;
						// fprintf(stderr, " %zu", chidx);
					}
					// fprintf(stderr, "  n_c_a: %zu->%zu\n",
					//         num_chunks_available, num_chunks_available + s.chunks.size());
					num_chunks_available += s.chunks.size();
					CUSUC(cudaStreamDestroy(s.stream));

					// fprintf(stderr, "-- revdeps = {");
					// for (job_t j : info.revdeps) fprintf(stderr, "%d,", j);
					// fprintf(stderr, "}\n");

					std::vector<job_t> new_sources;

					for (job_t rd : info.revdeps) {
						struct job_info_t &info2 = jobmap.find(rd)->second;

						// fprintf(stderr, "-- deps(%d) = {", rd);
						// for (job_t j : info2.deps) fprintf(stderr, "%d,", j);
						// fprintf(stderr, "}\n");

						auto it = std::find(info2.deps.begin(), info2.deps.end(), s.job);
						info2.deps.erase(it);
						if (info2.deps.size() == 0) {
							new_sources.push_back(rd);
						}
					}

					source_jobs.insert(source_jobs.begin(), new_sources.begin(), new_sources.end());

					jobmap.erase(jobmap.find(s.job));

					running_streams.erase(running_streams.begin() + i);
					i--;

					was_ready = true;
					continue;
				} else if (err == cudaErrorNotReady) {
					continue;
				} else {
					CUSUC(err);
				}
			}

			if (!was_ready) usleep(1000);
			continue;
		}

		// fprintf(stderr, "-- source_jobs =");
		// for (job_t j : source_jobs) fprintf(stderr, " %d(%dc)", j, jobmap.find(j)->second.num_chunks);
		// fprintf(stderr, "\n");
		// fprintf(stderr, "-- %zu chunks available\n", num_chunks_available);

		fprintf(stderr, "-- Starting job %d\n", id);

		source_jobs.erase(source_jobs.begin() + index);

		struct job_info_t &info = jobmap.find(id)->second;

		struct stream_info_t sinfo;
		sinfo.job = id;
		CUSUC(cudaStreamCreate(&sinfo.stream));

		struct job_data_t data;
		data.stream = sinfo.stream;
		for (size_t i = 0; i < devChunks.size() && data.chunks.size() < info.num_chunks; i++) {
			if (!chunk_taken[i]) {
				data.chunks.push_back(devChunks[i]);
				sinfo.chunks.push_back(i);
				chunk_taken[i] = true;
			}
		}
		assert(data.chunks.size() == info.num_chunks);
		num_chunks_available -= data.chunks.size();

		sinfo.data = data;

		info.pre_callback(data);

		running_streams.push_back(sinfo);

		total_started++;
	}

	for (void *ptr : devChunks) {
		CUSUC(cudaFree(ptr));
	}

	if (jobmap.size() == 0) {
		fprintf(stderr, "-- All jobs ran\n");
	} else {
		fprintf(stderr, "-- WARNING: NOT ALL JOBS RAN!\n");
	}
}

// vim: set sw=4 ts=4 noet:
