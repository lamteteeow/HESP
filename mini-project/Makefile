# Makefile für rigid_body CUDA-Projekt ohne Objektdateien

# Pfad zur CUDA-Installation (ggf. anpassen)
CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

# Zielname
TARGET    := rigid_body

# Include-Pfade
INCLUDES  := -I.

# nvcc-Flags: C++20 + separable compilation
NVCCFLAGS := -std=c++20 -rdc=true

.PHONY: all clean

all:
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) main.cu -o $(TARGET)

clean:
	rm -f $(TARGET)
