# Compiler and flags
NVCC = nvcc
CFLAGS = -x cu -std=c++17 --extended-lambda -O3
# -m64
LDFLAGS = -L./lib/x64 -I./include -lglfw3 -lglew32 -lopengl32 -lgdi32 -luser32 -lmsvcrt -Xlinker /NODEFAULTLIB:LIBCMT
#  -lkernel32

# Binary and source
BIN = moly.exe
SRC = moldyn.cu

# Default target
all: compile

# Compilation target
compile:
	$(NVCC) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

# Run target
run:
	$(BIN)

# Clean build artifacts
clean:
	rm -f $(BIN) *.o *.obj *.exp *.lib

# Phony targets
.PHONY: all compile run clean
