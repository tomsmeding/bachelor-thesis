#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include "topo.h"


static struct topology topo_cache;
static bool cache_filled = false;


struct topology get_topology(void) {
	if (cache_filled) return topo_cache;

	struct topology topo;
	memset(&topo, 0, sizeof topo);

	bool core_seen[1024];
	memset(core_seen, 0, sizeof core_seen);

	char fname[60];
	for (topo.nthreads = 0; ; topo.nthreads++) {
		snprintf(fname, sizeof fname, "/sys/devices/system/cpu/cpu%d/topology/core_id", topo.nthreads);
		int fd = open(fname, O_RDONLY);
		if (fd == -1) break;

		char buf[128];
		ssize_t nr = read(fd, buf, sizeof buf);
		close(fd);

		if (nr < 1) break;

		while (nr > 0 && isspace(buf[nr - 1])) nr--;
		buf[nr] = '\0';

		char *endp;
		int id = strtol(buf, &endp, 10);
		if (endp - buf != nr) {
			fprintf(stderr, "topo.c: Invalid core_id found for cpu%d\n", topo.nthreads);
			break;
		}

		assert(id < (int)(sizeof core_seen / sizeof core_seen[0]));
		core_seen[id] = true;
	}

	for (int i = 0; i < (int)(sizeof core_seen / sizeof core_seen[0]); i++) {
		if (core_seen[i]) topo.ncores++;
	}

	topo_cache = topo;
	cache_filled = true;

	return topo;
}

// vim: set sw=4 ts=4 noet:
