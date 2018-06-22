#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include "job.h"
#include "topo.h"


struct job_info_t {
	job_t id;
	void (*callback)(void*);
	void *callback_data;
	struct job_dep_list_t deps, revdeps;
};

struct jobmap_bucket_t {
	int num, cap;
	struct job_info_t **infos;
};

#define JOBMAP_NBUCKETS 256

static struct jobmap_bucket_t jobmap[JOBMAP_NBUCKETS];

static int source_jobs_cap = 0, source_jobs_num = 0;
static job_t *source_jobs = NULL;


__attribute__((constructor))
static void source_jobs_init(void) {
	source_jobs_cap = 64;
	source_jobs_num = 0;
	source_jobs = malloc(source_jobs_cap * sizeof(job_t));
}

__attribute__((destructor))
static void source_jobs_free(void) {
	free(source_jobs);
	source_jobs = NULL;
	source_jobs_cap = source_jobs_num = 0;
}

static void source_jobs_append(job_t id) {
	if (source_jobs_num + 1 > source_jobs_cap) {
		source_jobs_cap *= 2;
		source_jobs = realloc(source_jobs, source_jobs_cap * sizeof(job_t));
	}
	source_jobs[source_jobs_num++] = id;
}

static void source_jobs_find_remove(job_t id) {
	for (int i = 0; i < source_jobs_num; i++) {
		if (source_jobs[i] == id) {
			memmove(source_jobs + i, source_jobs + i + 1,
					(source_jobs_num - i - 1) * sizeof(struct job_info_t*));
			source_jobs_num--;
			return;
		}
	}
	fprintf(stderr, "Could not find id %d in source_jobs!\n", id);
	assert(false);
}

__attribute__((constructor))
static void jobmap_init(void) {
	for (int i = 0; i < JOBMAP_NBUCKETS; i++) {
		jobmap[i].cap = jobmap[i].num = 0;
		jobmap[i].infos = NULL;
	}
}

__attribute__((destructor))
static void jobmap_free(void) {
	for (int i = 0; i < JOBMAP_NBUCKETS; i++) {
		for (int j = 0; j < jobmap[i].num; j++) {
			job_dep_list_free(jobmap[i].infos[j]->deps);
			job_dep_list_free(jobmap[i].infos[j]->revdeps);
			free(jobmap[i].infos[j]);
		}
		if (jobmap[i].infos) free(jobmap[i].infos);
	}
}

static int jobmap_size(void) {
	int size = 0;
	for (int i = 0; i < JOBMAP_NBUCKETS; i++) {
		size += jobmap[i].num;
	}
	return size;
}

static struct job_info_t* jobmap_find(job_t id) {
	struct jobmap_bucket_t *bucket = &jobmap[id % JOBMAP_NBUCKETS];
	for (int i = 0; i < bucket->num; i++) {
		if (bucket->infos[i]->id == id) {
			return bucket->infos[i];
		}
	}
	fprintf(stderr, "Could not find id %d in jobmap!\n", id);
	assert(false);
}

static void jobmap_find_remove(job_t id) {
	struct jobmap_bucket_t *bucket = &jobmap[id % JOBMAP_NBUCKETS];
	for (int i = 0; i < bucket->num; i++) {
		if (bucket->infos[i]->id == id) {
			job_dep_list_free(bucket->infos[i]->deps);
			job_dep_list_free(bucket->infos[i]->revdeps);
			free(bucket->infos[i]);
			memmove(bucket->infos + i, bucket->infos + i + 1,
					(bucket->num - i - 1) * sizeof(struct jobmap_bucket_t*));
			bucket->num--;
			return;
		}
	}
	fprintf(stderr, "Could not find id %d in jobmap!\n", id);
	assert(false);
}

static void jobmap_add(struct job_info_t *info) {
	struct jobmap_bucket_t *bucket = &jobmap[info->id % JOBMAP_NBUCKETS];

	if (bucket->cap == 0) {
		bucket->cap = 2;
		bucket->num = 0;
		bucket->infos = malloc(bucket->cap * sizeof(struct job_info_t*));
	}

	if (bucket->num + 1 > bucket->cap) {
		bucket->cap *= 2;
		bucket->infos = realloc(bucket->infos, bucket->cap * sizeof(struct job_info_t*));
	}
	bucket->infos[bucket->num++] = info;
}


static job_t gen_job_id(void) {
	static job_t id = 0;
	return id++;
}


struct job_dep_list_t job_dep_list(int ndeps, ...) {
	assert(ndeps >= 0);

	va_list ap;
	va_start(ap, ndeps);

	struct job_dep_list_t list;
	list.cap = ndeps;
	list.num = 0;
	list.list = list.cap == 0 ? NULL : malloc(list.cap * sizeof(job_t));

	for (int i = 0; i < ndeps; i++) {
		job_t id = va_arg(ap, job_t);
		if (id != -1) list.list[list.num++] = id;
	}

	va_end(ap);

	return list;
}

struct job_dep_list_t job_dep_merge(int ndeps, ...) {
	assert(ndeps >= 0);

	va_list ap, ap2;
	va_start(ap, ndeps);
	va_copy(ap2, ap);

	int total = 0;
	for (int i = 0; i < ndeps; i++) {
		total += va_arg(ap2, struct job_dep_list_t*)->num;
	}

	va_end(ap2);

	struct job_dep_list_t list;
	list.cap = total;
	list.num = total;
	list.list = list.cap == 0 ? NULL : malloc(list.cap * sizeof(job_t));

	int cursor = 0;
	for (int i = 0; i < ndeps; i++) {
		struct job_dep_list_t *a = va_arg(ap, struct job_dep_list_t*);
		memcpy(list.list + cursor, a->list, a->num * sizeof(job_t));
		cursor += a->num;
	}
	assert(cursor == list.num);

	va_end(ap);

	return list;
}

void job_dep_list_free(struct job_dep_list_t list) {
	// free() already checks for NULL
	free(list.list);
}

void job_dep_list_append(struct job_dep_list_t *list, job_t id) {
	if (id == -1) return;
	if (list->num + 1 > list->cap) {
		if (list->cap == 0) list->cap = 2;
		else list->cap *= 2;
		list->list = realloc(list->list, list->cap * sizeof(job_t));
	}
	list->list[list->num++] = id;
}

void job_dep_list_append_list(struct job_dep_list_t *list, struct job_dep_list_t list2) {
	job_dep_list_append_listp(list, &list2);
	job_dep_list_free(list2);
}

void job_dep_list_append_listp(struct job_dep_list_t *list, struct job_dep_list_t *list2) {
	if (list->num + list2->num > list->cap) {
		if (list->cap == 0) list->cap = 2;
		else do list->cap *= 2; while (list->num + list2->num > list->cap);
		list->list = realloc(list->list, list->cap * sizeof(job_t));
	}
	memcpy(list->list + list->num, list2->list, list2->num * sizeof(job_t));
	list->num += list2->num;
}

void job_dep_list_remove(struct job_dep_list_t *list, job_t id) {
	if (id == -1) return;
	for (int i = 0; i < list->num; i++) {
		if (list->list[i] == id) {
			memmove(list->list + i, list->list + i + 1, (list->num - i - 1) * sizeof(job_t));
			list->num--;
			break;
		}
	}
}

job_t job_submit(struct job_dep_list_t deps, void (*callback)(void*), void *callback_data) {
	job_t id = gen_job_id();

	struct job_info_t *info = malloc(sizeof(struct job_info_t));
	info->id = id;
	info->callback = callback;
	info->callback_data = callback_data;
	info->deps = deps;
	info->revdeps = job_dep_list(0);

	// fprintf(stderr, "job_submit: id=%d deps=", id);
	// for (int i = 0; i < info->deps.num; i++) fprintf(stderr, "%d,", info->deps.list[i]);
	// fprintf(stderr, "\n");

	for (int i = 0; i < info->deps.num; i++) {
		job_t dep = info->deps.list[i];
		struct job_info_t *info2 = jobmap_find(dep);
		job_dep_list_append(&info2->revdeps, id);
	}

	if (info->deps.num == 0) {
		source_jobs_append(id);
	}

	jobmap_add(info);

	return id;
}

void job_add_dep(job_t id, job_t dep) {
	if (id == -1 || dep == -1) return;
	struct job_info_t *info = jobmap_find(id);
	if (info->deps.num == 0) {
		source_jobs_find_remove(id);
	}
	job_dep_list_append(&info->deps, dep);
	struct job_info_t *info2 = jobmap_find(dep);
	job_dep_list_append(&info2->revdeps, id);
}

void job_add_dep_list(job_t id, struct job_dep_list_t deps) {
	if (id == -1) return;
	job_add_dep_listp(id, &deps);
	job_dep_list_free(deps);
}

void job_add_dep_listp(job_t id, struct job_dep_list_t *deps) {
	if (id == -1) return;
	struct job_info_t *info = jobmap_find(id);
	if (info->deps.num == 0) {
		source_jobs_find_remove(id);
	}
	for (int i = 0; i < deps->num; i++) {
		job_t dep = deps->list[i];
		struct job_info_t *info2 = jobmap_find(dep);
		job_dep_list_append(&info2->revdeps, id);
	}
	job_dep_list_append_listp(&info->deps, deps);
}

void job_add_dep_list_list(struct job_dep_list_t *ids, struct job_dep_list_t deps) {
	job_add_dep_list_listp(ids, &deps);
	job_dep_list_free(deps);
}

void job_add_dep_list_listp(struct job_dep_list_t *ids, struct job_dep_list_t *deps) {
	for (int i = 0; i < ids->num; i++) {
		job_t id = ids->list[i];
		job_add_dep_listp(id, deps);
	}
}

void job_run_linear(void) {
	int total_njobs = jobmap_size();
	fprintf(stderr, "-- Linear job run, %d job%s\n", total_njobs, "s" + (total_njobs == 1));

	while (source_jobs_num > 0) {
		// fprintf(stderr, "-- source_jobs = {");
		// for (int i = 0; i < source_jobs_num; i++)
		//     fprintf(stderr, "%d,", source_jobs[i]);
		// fprintf(stderr, "}\n");

		job_t id = source_jobs[0];
		memmove(source_jobs, source_jobs + 1, (source_jobs_num - 1) * sizeof(job_t));
		source_jobs_num--;

		fprintf(stderr, "-- Running job %d\n", id);

		struct job_info_t *info = jobmap_find(id);
		info->callback(info->callback_data);

		// fprintf(stderr, "-- revdeps = {");
		// for (int i = 0; i < info->revdeps.num; i++)
		//     fprintf(stderr, "%d,", info->revdeps.list[i]);
		// fprintf(stderr, "}\n");

		struct job_dep_list_t new_sources = job_dep_list(0);

		for (int i = 0; i < info->revdeps.num; i++) {
			job_t rd = info->revdeps.list[i];
			struct job_info_t *info2 = jobmap_find(rd);

			// fprintf(stderr, "-- deps(%d) = {", rd);
			// for (int i = 0; i < info2->deps.num; i++)
			//     fprintf(stderr, "%d,", info2->deps.list[i]);
			// fprintf(stderr, "}\n");

			for (int i = 0; i < info2->deps.num; i++) {
				if (info2->deps.list[i] == id) {
					memmove(info2->deps.list + i, info2->deps.list + i + 1,
							(info2->deps.num - i - 1) * sizeof(job_t));
					info2->deps.num--;
					break;
				}
			}
			if (info2->deps.num == 0) {
				job_dep_list_append(&new_sources, rd);
			}
		}

		if (source_jobs_num + new_sources.num > source_jobs_cap) {
			do source_jobs_cap *= 2;
			while (source_jobs_num + new_sources.num > source_jobs_cap);
			source_jobs = realloc(source_jobs, source_jobs_cap * sizeof(job_t));
		}
		memmove(source_jobs + new_sources.num, source_jobs,
				source_jobs_num * sizeof(job_t));
		memcpy(source_jobs, new_sources.list, new_sources.num * sizeof(job_t));
		source_jobs_num += new_sources.num;

		job_dep_list_free(new_sources);

		jobmap_find_remove(id);
	}

	if (jobmap_size() == 0) {
		fprintf(stderr, "-- All jobs ran\n");
	} else {
		fprintf(stderr, "-- WARNING: NOT ALL JOBS RAN!\n");
	}
}

// Job Concurrent Debug
// #define JCDEBUG

//   info   | busy  |
// ---------+-------+-------
// NULL     | false | Idle, waiting for new job
// non-NULL | false | Recently completed job, waiting for control to visit
// non-NULL | true  | Busy processing a job
struct worker_t {
	int worker_id;

	pthread_t th;
	pthread_mutex_t mutex_newwork;
	pthread_cond_t cond_newwork;
	bool busy;

	pthread_mutex_t *mutex_control;
	pthread_cond_t *cond_control;

	// Current job to execute
	const struct job_info_t *info;
};

static void* worker_entry(void *data_) {
	struct worker_t *data = (struct worker_t*)data_;

	pthread_mutex_lock(&data->mutex_newwork);

#ifdef JCDEBUG
	fprintf(stderr, "[W%d] Created\n", data->worker_id);
#endif

	while (true) {
		const struct job_info_t *current_info = data->info;

		if (current_info != NULL) {
#ifdef JCDEBUG
			fprintf(stderr, "[W%d] Starting job %d\n", data->worker_id, current_info->id);
#endif

			current_info->callback(current_info->callback_data);

#ifdef JCDEBUG
			fprintf(stderr, "[W%d] Completed job %d\n", data->worker_id, current_info->id);
#endif

			pthread_mutex_lock(data->mutex_control);

			data->busy = false;
			pthread_cond_signal(data->cond_control);

			pthread_mutex_unlock(data->mutex_control);
		}

		pthread_cond_wait(&data->cond_newwork, &data->mutex_newwork);

#ifdef JCDEBUG
		fprintf(stderr, "[W%d] New work signalled, let's look...\n", data->worker_id);
#endif
	}

	// Should be unreachable
	assert(false);
}

// TODO: pthreads error checking
void job_run_concur(int ncores) {
	struct topology topology = get_topology();

	/*printf("digraph G {\n");
	for (int i = 0; i < JOBMAP_NBUCKETS; i++) {
		for (int j = 0; j < jobmap[i].num; j++) {
			struct job_info_t *info = jobmap[i].infos[j];
			for (int k = 0; k < info->deps.num; k++) {
				printf("\t%d -> %d;\n", info->deps.list[k], info->id);
			}
			if (info->deps.num == 0) {
				printf("\t%d [color=red];\n", info->id);
			}
			// for (int k = 0; k < info->revdeps.num; k++) {
			//     printf("\t%d -> %d [color=red];\n", info->id, info->revdeps.list[k]);
			// }
		}
	}
	printf("}\n");
	exit(0);*/
	int seed = time(NULL);
	fprintf(stderr, "SEED = %d\n", seed);
	srand(seed);

	const int num_workers = ncores;
	int total_njobs = jobmap_size();
	fprintf(stderr, "-- Concurrent job run, %d job%s\n", total_njobs, "s" + (total_njobs == 1));
	fprintf(stderr, "-- Using %d workers (%d threads, %d cores)\n", num_workers, topology.nthreads, topology.ncores);

	pthread_mutex_t mutex_control;
	pthread_cond_t cond_control;
	pthread_mutex_init(&mutex_control, NULL);
	pthread_cond_init(&cond_control, NULL);

	pthread_mutex_lock(&mutex_control);

	struct worker_t workers[num_workers];

	for (int i = 0; i < num_workers; i++) {
		workers[i].worker_id = i;
		workers[i].mutex_control = &mutex_control;
		workers[i].cond_control = &cond_control;
		workers[i].busy = false;
		workers[i].info = NULL;

		pthread_mutex_init(&workers[i].mutex_newwork, NULL);
		pthread_cond_init(&workers[i].cond_newwork, NULL);

		pthread_create(&workers[i].th, NULL, worker_entry, &workers[i]);
	}

	while (true) {
		// fprintf(stderr, "-- source_jobs = {");
		// for (int i = 0; i < source_jobs_num; i++)
		//     fprintf(stderr, "%d,", source_jobs[i]);
		// fprintf(stderr, "}\n");
		
		int target_worker;
		for (target_worker = 0; target_worker < num_workers; target_worker++) {
			if (workers[target_worker].info == NULL) break;
		}

		if (source_jobs_num > 0 && target_worker < num_workers) {
			// We have a worker that is ready for a new job, and a job that is
			// ready to execute. Match them.

			// int index = rand() % source_jobs_num;
			int index = 0;
			job_t id = source_jobs[index];
			memmove(source_jobs + index, source_jobs + index + 1, (source_jobs_num - index - 1) * sizeof(job_t));
			source_jobs_num--;

#ifdef JCDEBUG
			fprintf(stderr, "-- Running job %d on worker %d\n", id, target_worker);
#endif

			pthread_mutex_lock(&workers[target_worker].mutex_newwork);

			workers[target_worker].info = jobmap_find(id);
			workers[target_worker].busy = true;
			pthread_cond_signal(&workers[target_worker].cond_newwork);

			pthread_mutex_unlock(&workers[target_worker].mutex_newwork);

			continue;
		}

		// Either no worker is ready to take a job, or there are no ready jobs.
		// If all workers are idle, this means that we're done and should exit.
		// Otherwise, we should wait until some dependency-less job becomes
		// available.

		bool all_idle = true, any_completed = false;
		for (int i = 0; i < num_workers; i++) {
			if (workers[i].info == NULL) continue;

			if (workers[i].busy) {
				all_idle = false;
			} else {
				// This one is done, handle its completion. Afterwards it will
				// be idle, so we don't reset all_idle here.

				const struct job_info_t *info = workers[i].info;

#ifdef JCDEBUG
				fprintf(stderr, "-- Worker %d completed job %d\n", i, info->id);
#endif

				struct job_dep_list_t new_sources = job_dep_list(0);

				for (int i = 0; i < info->revdeps.num; i++) {
					job_t rd = info->revdeps.list[i];
					struct job_info_t *info2 = jobmap_find(rd);

					job_dep_list_remove(&info2->deps, info->id);
					if (info2->deps.num == 0) {
						job_dep_list_append(&new_sources, rd);
					}
				}

				if (source_jobs_num + new_sources.num > source_jobs_cap) {
					do source_jobs_cap *= 2;
					while (source_jobs_num + new_sources.num > source_jobs_cap);
					source_jobs = realloc(source_jobs, source_jobs_cap * sizeof(job_t));
				}
#if 1
				memmove(source_jobs + new_sources.num, source_jobs,
						source_jobs_num * sizeof(job_t));
				memcpy(source_jobs, new_sources.list, new_sources.num * sizeof(job_t));
#else
				memcpy(source_jobs + source_jobs_num, new_sources.list, new_sources.num * sizeof(job_t));
#endif
				source_jobs_num += new_sources.num;

				job_dep_list_free(new_sources);

				jobmap_find_remove(info->id);

				workers[i].info = NULL;

				any_completed = true;
			}
		}

		// If we handled completion for some thread, we should give them work
		// again immediately, so skip to the beginning of the loop. Note that
		// this is necessary even if all threads are now idle, since we might
		// have gained new source jobs by handling some completions.
		if (any_completed) continue;

		// Otherwise, if all were idle, we apparently didn't have any work to
		// give out anymore, so we're done.
		if (all_idle) {
#ifdef JCDEBUG
			fprintf(stderr, "-- All workers found idle, done\n");
#endif
			break;
		}

		while (true) {
			int total_busy = 0;
			for (int i = 0; i < num_workers; i++) total_busy += workers[i].busy;
#ifdef JCDEBUG
			fprintf(stderr, "-- Waiting for worker to complete... (%d busy)\n", total_busy);
#endif

			pthread_cond_wait(&cond_control, &mutex_control);

			bool any_done = false;
			for (int i = 0; i < num_workers; i++) {
				if (workers[i].info != NULL && workers[i].busy == false) {
					any_done = true;
#ifdef JCDEBUG
					fprintf(stderr, "-- Worker %d completed a job\n", i);
#endif
					break;
				}
			}

			if (any_done) break;
		}
	}

	for (int i = 0; i < num_workers; i++) {
		pthread_cancel(workers[i].th);
		void *res;
		pthread_join(workers[i].th, &res);
		assert(res == PTHREAD_CANCELED);

		pthread_mutex_destroy(&workers[i].mutex_newwork);
		pthread_cond_destroy(&workers[i].cond_newwork);
	}

	pthread_mutex_unlock(&mutex_control);
	pthread_mutex_destroy(&mutex_control);
	pthread_cond_destroy(&cond_control);

	if (jobmap_size() == 0) {
		fprintf(stderr, "-- All jobs ran\n");
	} else {
		fprintf(stderr, "-- WARNING: NOT ALL JOBS RAN!\n");
	}
}

// vim: set sw=4 ts=4 noet:
