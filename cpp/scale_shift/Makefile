# Local build when cmake is not on PATH (this login node).
# CI uses CMakePresets (`cmake --preset ci`).

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude -fopenmp
LDFLAGS ?= -fopenmp

SRC = src/npy.cpp src/fit_scale_shift.cpp
OBJ = $(SRC:.cpp=.o)

.PHONY: all test clean

all: scale_shift_fit smoke_test

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

scale_shift_fit: $(OBJ) src/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

src/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

smoke_test: $(OBJ) tests/smoke.cpp
	$(CXX) $(CXXFLAGS) tests/smoke.cpp $(OBJ) -o $@ $(LDFLAGS)

test: smoke_test
	./smoke_test

clean:
	rm -f $(OBJ) src/main.o scale_shift_fit smoke_test
