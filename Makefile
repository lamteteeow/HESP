# Makefile for md2d — GPU-accelerated 2D molecular dynamics simulator
#
# Usage:
#   make              — build the md2d binary
#   make clean        — remove build artifacts
#
# On the TinyGPU cluster, load CUDA before building:
#   module load cuda
#   make

# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
NVCC := nvcc

# CUDA toolkit path (set manually if nvcc is not in PATH)
CUDA_HOME ?=

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

# Language standard (C++17 for both host and device)
NVCCFLAGS := -std=c++17

# Optimisation
NVCCFLAGS += -O3

# Include paths
NVCCFLAGS += -I src

# Host compiler flags — portable across Intel and AMD CPUs (TinyGPU cluster)
# Uses x86-64-v3 baseline: AVX2 + FMA, safe on all TinyGPU partitions.
NVCCFLAGS += -Xcompiler=-mavx2 -Xcompiler=-mfma

# Multi-architecture GPU code generation (TinyGPU cluster GPUs)
NVCCFLAGS += -gencode arch=compute_70,code=sm_70   # Tesla V100
NVCCFLAGS += -gencode arch=compute_75,code=sm_75   # RTX 2080 Ti
NVCCFLAGS += -gencode arch=compute_80,code=sm_80   # A100
NVCCFLAGS += -gencode arch=compute_86,code=sm_86   # RTX 3080

# Relocatable device code (consistent with original CMake build)
# Suppress deprecation warning for sm_70 (V100) — still supported, just
# flagged as legacy in CUDA 12.8+. Remove this line once sm_70 is dropped.
NVCCFLAGS += -Wno-deprecated-gpu-targets

# Relocatable device code (consistent with original build)
NVCCFLAGS += -rdc=true

# Linker flags
LDFLAGS := -rdc=true

# ---------------------------------------------------------------------------
# Source and targets
# ---------------------------------------------------------------------------
SRC        := src/main.cu
TARGET_2D  := md2d

# 3D build: add -DMD3D to enable third dimension
NVCCFLAGS_3D := $(NVCCFLAGS) -DMD3D
TARGET_3D    := md3d

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
.PHONY: all clean help md3d

all: $(TARGET_2D)

$(TARGET_2D): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC) $(LDFLAGS)

md3d: $(SRC)
	$(NVCC) $(NVCCFLAGS_3D) -o $(TARGET_3D) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET_2D) $(TARGET_3D)

help:
	@echo "md2d / md3d — GPU-accelerated 2D/3D molecular dynamics simulator"
	@echo ""
	@echo "Targets:"
	@echo "  all       build md2d (2D binary, default)"
	@echo "  md3d      build md3d (3D binary, defines MD3D)"
	@echo "  clean     remove binaries"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_HOME     path to CUDA toolkit (default: auto-detect from PATH)"
	@echo "  NVCC          CUDA compiler  (default: nvcc)"
	@echo ""
	@echo "On TinyGPU cluster, first load the CUDA module:"
	@echo "  module load cuda"
	@echo "  make"
