#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace spur {

// Little-endian float32 C-order 2-D (or HxWx1) .npy.
struct Array2f {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;  // row-major

  float operator()(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
  float& operator()(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
};

Array2f read_npy_f32(const std::filesystem::path& path);
void write_npy_f32(const std::filesystem::path& path, const Array2f& arr);

}  // namespace spur
