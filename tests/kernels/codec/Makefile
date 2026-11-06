# SPDX-License-Identifier: Apache-2.0
# Standalone codec test drivers (the library build lives in the top-level CMake).
#   make ARCH=sm_110            # Thor
#   make ARCH=sm_120            # RTX PRO 6000
#   make all-arch ALL_GENCODE='...'  # compile-check the supplied -gencode targets
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_110
ROOT := ../../..
FLAGS := -std=c++17 -O3 -I$(ROOT)/csrc -I$(ROOT)/include -Xcompiler -fopenmp,-fPIC
SRCS := $(ROOT)/csrc/codec/encode.cu $(ROOT)/csrc/codec/decode.cu
HDRS := $(ROOT)/csrc/codec/codec.h $(ROOT)/csrc/codec/tile.cuh $(ROOT)/include/nf12/format.cuh $(ROOT)/csrc/internal.h

all: codec_test libcodec_shim.so

codec_test: codec_test.cu $(SRCS) $(HDRS)
	$(NVCC) $(FLAGS) -arch=$(ARCH) -o $@ codec_test.cu $(SRCS) -lgomp

libcodec_shim.so: codec_shim.cu $(SRCS) $(HDRS)
	$(NVCC) $(FLAGS) -arch=$(ARCH) -shared -o $@ codec_shim.cu $(SRCS)

all-arch: $(SRCS) $(HDRS)
	$(NVCC) $(FLAGS) $(ALL_GENCODE) -c $(ROOT)/csrc/codec/encode.cu -o /dev/null
	$(NVCC) $(FLAGS) $(ALL_GENCODE) -c $(ROOT)/csrc/codec/decode.cu -o /dev/null

clean:
	rm -f codec_test libcodec_shim.so

.PHONY: all all-arch clean
