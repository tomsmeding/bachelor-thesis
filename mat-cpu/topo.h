#pragma once


struct topology {
	int ncores, nthreads;
};

struct topology get_topology(void);

// vim: set sw=4 ts=4 noet:
