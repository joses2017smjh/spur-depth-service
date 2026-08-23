#include "spur/npy.hpp"

#include <cstring>
#include <fstream>
#include <sstream>

namespace spur {
namespace {

constexpr char kMagic[] = "\x93NUMPY";

void check(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

}  // namespace

Array2f read_npy_f32(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  check(in.good(), "cannot open " + path.string());
  char magic[6];
  in.read(magic, 6);
  check(in.gcount() == 6 && std::memcmp(magic, kMagic, 6) == 0, path.string() + ": not a .npy file");
  unsigned char ver[2];
  in.read(reinterpret_cast<char*>(ver), 2);
  std::uint32_t header_len = 0;
  if (ver[0] == 1) {
    std::uint16_t hl = 0;
    in.read(reinterpret_cast<char*>(&hl), 2);
    header_len = hl;
  } else {
    in.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  check(header.find("<f4") != std::string::npos || header.find("|f4") != std::string::npos,
        path.string() + ": expected little-endian float32");
  check(header.find("fortran_order': True") == std::string::npos &&
            header.find("fortran_order\": True") == std::string::npos,
        path.string() + ": fortran_order arrays are not supported");

  auto shape_pos = header.find("shape");
  check(shape_pos != std::string::npos, path.string() + ": missing shape");
  auto lpar = header.find('(', shape_pos);
  auto rpar = header.find(')', lpar);
  std::string inside = header.substr(lpar + 1, rpar - lpar - 1);
  std::vector<int> dims;
  std::stringstream ss(inside);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    // trim
    auto a = tok.find_first_not_of(" \t");
    auto b = tok.find_last_not_of(" \t");
    if (a == std::string::npos) continue;
    dims.push_back(std::stoi(tok.substr(a, b - a + 1)));
  }
  check(!dims.empty() && dims.size() <= 3, path.string() + ": expected 2-D (or HxWx1) array");
  if (dims.size() == 3) {
    check(dims[2] == 1, path.string() + ": 3-D last dim must be 1");
  }
  if (dims.size() == 1) {
    dims.push_back(1);
  }
  Array2f arr;
  arr.rows = dims[0];
  arr.cols = dims[1];
  arr.data.resize(static_cast<size_t>(arr.rows) * arr.cols);
  in.read(reinterpret_cast<char*>(arr.data.data()),
          static_cast<std::streamsize>(arr.data.size() * sizeof(float)));
  check(in.good() || in.eof(), path.string() + ": short read");
  return arr;
}

void write_npy_f32(const std::filesystem::path& path, const Array2f& arr) {
  std::ostringstream hdr;
  hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << arr.rows << ", " << arr.cols
      << "), }";
  std::string header = hdr.str();
  // v1 header: magic(6)+ver(2)+len(2)+header, padded to 16-byte multiple, newline-terminated.
  int prefix = 6 + 2 + 2;
  int pad = 16 - static_cast<int>((prefix + header.size() + 1) % 16);
  if (pad == 16) pad = 0;
  header.append(static_cast<size_t>(pad), ' ');
  header.push_back('\n');
  std::uint16_t hl = static_cast<std::uint16_t>(header.size());

  std::ofstream out(path, std::ios::binary);
  check(out.good(), "cannot write " + path.string());
  out.write(kMagic, 6);
  unsigned char ver[2] = {1, 0};
  out.write(reinterpret_cast<char*>(ver), 2);
  out.write(reinterpret_cast<char*>(&hl), 2);
  out.write(header.data(), header.size());
  out.write(reinterpret_cast<const char*>(arr.data.data()),
            static_cast<std::streamsize>(arr.data.size() * sizeof(float)));
}

}  // namespace spur
