CXX      = g++
CXXFLAGS = -O2 -Wall
# main() declares several large fixed-size arrays (e.g. rydprm[6][nrydmax], ~4.8 MB)
# as automatic/stack variables, which overflows the 1 MB default stack MinGW/Windows
# binaries get. Reserve a bigger stack at link time (MinGW-specific flag) rather than
# restructuring the arrays. Not needed on Linux/macOS, where the default stack
# (usually 8 MB) is already large enough.
ifeq ($(OS),Windows_NT)
LDFLAGS  = -Wl,--stack,16777216
endif

TARGET  = plasma_expansion
SRC     = plasma_expansion.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) $(TARGET).exe
