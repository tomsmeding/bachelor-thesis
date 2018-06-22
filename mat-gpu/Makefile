# CC = gcc
# CFLAGS = -Wall -Wextra -std=c11 -O2 -march=native -mtune=native -g
# CFLAGS += -Wno-sign-compare -Wno-unused-value -Wno-unused-variable

NVCC := nvcc --gpu-architecture=sm_61
CXX := g++
# NVCC := clang++ --cuda-gpu-arch=sm_61 --cuda-path=/opt/cuda -nocudalib
# CXX := clang++
CUFLAGS := -g -lineinfo -O3 -std=c++11
CXXFLAGS := -g -O3 -Wall -Wextra -std=c++11
LDFLAGS :=

TARGETS := main

CXX_OBJECTS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
CU_OBJECTS := $(patsubst %.cu,%.o,$(wildcard *.cu))
OBJECTS := $(CXX_OBJECTS) $(CU_OBJECTS)

ifneq ($(value EXTRA_FLAGS),)
	CUFLAGS := $(CUFLAGS) $(value EXTRA_FLAGS)
	CXXFLAGS := $(CXXFLAGS) $(value EXTRA_FLAGS)
endif


.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f $(TARGETS) *.o


main: $(OBJECTS) main.dev.o libgmp_asm_tuned.a
	$(NVCC) $(CUFLAGS) -o $@ $^ $(LDFLAGS)

main.dev.o: $(CU_OBJECTS)
	$(NVCC) $(CUFLAGS) -dlink -o $@ $^

$(CU_OBJECTS): %.o: %.cu $(wildcard *.h)
	$(NVCC) $(CUFLAGS) -dc -o $@ $<

$(CXX_OBJECTS): %.o: %.cpp $(wildcard *.h)
	$(CXX) $(CXXFLAGS) -c -o $@ $<
