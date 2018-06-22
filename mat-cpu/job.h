#pragma once


typedef int job_t;

struct job_dep_list_t {
	int num, cap;
	job_t *list;
};

// If a job_dep_list_t is taken by value, its ownership is transferred to the
// called function. If its pointer is taken, the ownership stays with the
// caller.

// A job_t value of -1 is taken to be a nonexistent job, so e.g.
// job_dep_list_append(list, -1) is a no-op.
// However, a job_dep_list_t should never contain -1's, which it will not when
// constructed using the functions here.

struct job_dep_list_t job_dep_list(int ndeps, ...);  // takes job_t's
struct job_dep_list_t job_dep_merge(int ndeps, ...);  // takes struct job_dep_list_t*'s
void job_dep_list_free(struct job_dep_list_t list);
void job_dep_list_append(struct job_dep_list_t *list, job_t id);
void job_dep_list_append_list(struct job_dep_list_t *list, struct job_dep_list_t list2);
void job_dep_list_append_listp(struct job_dep_list_t *list, struct job_dep_list_t *list2);
void job_dep_list_remove(struct job_dep_list_t *list, job_t id);

job_t job_submit(struct job_dep_list_t deps, void (*callback)(void*), void *callback_data);

void job_add_dep(job_t id, job_t dep);
void job_add_dep_list(job_t id, struct job_dep_list_t deps);
void job_add_dep_listp(job_t id, struct job_dep_list_t *deps);
void job_add_dep_list_list(struct job_dep_list_t *ids, struct job_dep_list_t deps);
void job_add_dep_list_listp(struct job_dep_list_t *ids, struct job_dep_list_t *deps);

void job_run_linear(void);
void job_run_concur(int ncores);

// vim: set sw=4 ts=4 noet:
