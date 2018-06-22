Code of the large-integer matrix multiplication programs, written for my
bachelor thesis at Leiden University in spring 2018.

There is a CPU implementation and a GPU implementation. Both have been tested
and work on Arch Linux and on Ubuntu 16.04. The code can be compiled using
`make` in the respective directories.

Both directories contain a file `libgmp_asm_tuned.a`, which has been compiled
for Arch Linux. To compile a specific one for your machine, please use GMP
version 6.1.2. Note that GMP supports machine-specific tuning using the bundled
`tuneup` program. The static library is generated in the normal GMP compilation
process.

The executables respond to the `-h` flag for some brief flag information; peruse
the source, especially main.c and main.cu for mat-cpu and mat-gpu, respectively,
for more information about exact usage.


The PDF of the actual thesis may be committed later.
