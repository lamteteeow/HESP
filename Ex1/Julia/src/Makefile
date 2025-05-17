# configuration

BUILD_DIR = ../build


# all

targets = $(BUILD_DIR)/julia


.PHONY: all
all: mk-target-dir $(targets)


mk-target-dir:
	mkdir -p $(BUILD_DIR)


# build rules

$(BUILD_DIR)/julia: julia.cu
	nvcc -O3 -std=c++17  -arch=sm_86 -o $@ $<


# aliases without build directory

.PHONY: julia
julia: $(BUILD_DIR)/julia


# clean target

.PHONY: clean
clean:
	rm -f $(targets)
