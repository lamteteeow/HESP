# Compiler and flags
NVCC = nvcc
CFLAGS = -x cu -std=c++17 --extended-lambda -O3
LDFLAGS = -lGL -lGLEW -lglfw -lm -ldl -lX11

# Binary and source
BIN = moly
SRC = moldyn.cu

# Default target
all: compile

# Compilation target
compile:
	$(NVCC) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

# Run target
run:
	./$(BIN)

# Clean build artifacts
clean:
	rm -f $(BIN) *.o

# Phony targets
.PHONY: all compile run clean
