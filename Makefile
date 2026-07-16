# Makefile for md3d — GPU-accelerated 3D DEM simulator
#
# Usage:
#   make              — build the md3d binary
#   make clean        — remove build artifacts
#
# On the TinyGPU cluster, load CUDA before building:
#   module load cuda
#   make

# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
NVCC := nvcc

CUDA_HOME ?=

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
NVCCFLAGS := -std=c++17 -O3 -I src

# Host compiler flags — portable across Intel and AMD CPUs (TinyGPU cluster)
NVCCFLAGS += -Xcompiler=-mavx2 -Xcompiler=-mfma

# Multi-architecture GPU code generation (TinyGPU cluster GPUs)
NVCCFLAGS += -gencode arch=compute_70,code=sm_70   # Tesla V100
NVCCFLAGS += -gencode arch=compute_75,code=sm_75   # RTX 2080 Ti
NVCCFLAGS += -gencode arch=compute_80,code=sm_80   # A100
NVCCFLAGS += -gencode arch=compute_86,code=sm_86   # RTX 3080

# Suppress legacy GPU target warnings
NVCCFLAGS += -Wno-deprecated-gpu-targets

# Suppress host/device annotation warning on defaulted constructors
NVCCFLAGS += -diag-suppress 20012

# Relocatable device code (required for separate compilation across .cu files)
NVCCFLAGS += -rdc=true
LDFLAGS := -rdc=true

# ---------------------------------------------------------------------------
# Source and targets
# ---------------------------------------------------------------------------
SRC := $(wildcard src/*.cu)
TARGET := md3d

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

help:
	@echo "md3d — Multi-GPU 3D DEM simulator"
	@echo ""
	@echo "Targets:"
	@echo "  make       build md3d"
	@echo "  make clean remove binary"
	@echo ""
	@echo "Usage:"
	@echo "  ./md3d <scene.json> [max_steps] [num_gpus] [vtk_interval] [bench_interval]"
	@echo ""
	@echo "On TinyGPU cluster, first load the CUDA module:"
	@echo "  module load cuda"
	@echo "  make"
