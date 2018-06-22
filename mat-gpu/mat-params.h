#pragma once


#define INTERLEAVE_BLOCK_SIZE 256
#define BASE_SIZE 256
#define TRANSFER_SIZE 512

// These values seem to be optimal for method 47 (interleaved naive)
#ifndef GPU_BLK_W
#define GPU_BLK_W 32
#endif
#ifndef GPU_BLK_H
#define GPU_BLK_H 2
#endif

// vim: set sw=4 ts=4 noet:
