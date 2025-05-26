CC = nvcc
CXX = g++
CXXFLAGS = -O3 -Wall
LDFLAGS = -lm

SRC_DIR = src
KERNELS_DIR = $(SRC_DIR)/kernels
UTILS_DIR = $(SRC_DIR)/utils
HEADERS_DIR = $(SRC_DIR)/headers
TESTS_DIR = tests

TARGET = particle_simulator
OBJECTS = $(SRC_DIR)/main.o $(SRC_DIR)/particle_sim.o $(KERNELS_DIR)/force_calculation.o $(KERNELS_DIR)/integration.o $(UTILS_DIR)/file_io.o $(UTILS_DIR)/vtk_writer.o $(UTILS_DIR)/config_parser.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cu $(HEADERS_DIR)/*.h
	$(CC) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/particle_sim.o: $(SRC_DIR)/particle_sim.cu $(HEADERS_DIR)/*.h
	$(CC) $(CXXFLAGS) -c $< -o $@

$(KERNELS_DIR)/force_calculation.o: $(KERNELS_DIR)/force_calculation.cu
	$(CC) $(CXXFLAGS) -c $< -o $@

$(KERNELS_DIR)/integration.o: $(KERNELS_DIR)/integration.cu
	$(CC) $(CXXFLAGS) -c $< -o $@

$(UTILS_DIR)/file_io.o: $(UTILS_DIR)/file_io.cpp $(HEADERS_DIR)/*.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(UTILS_DIR)/vtk_writer.o: $(UTILS_DIR)/vtk_writer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(UTILS_DIR)/config_parser.o: $(UTILS_DIR)/config_parser.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)