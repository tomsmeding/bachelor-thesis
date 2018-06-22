#!/usr/bin/env bash
rm -f main.o mat-gpu.o mat-gpu-launchers.o mat-pregpu.o
make "$@"
exit $?
