// Standalone smoke test for `make test` when GoogleTest/CMake are absent.
#include "spur/fit_scale_shift.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using spur::Array2f;
using spur::FitConfig;

static Array2f ramp(int h, int w) {
  Array2f a;
  a.rows = h;
  a.cols = w;
  a.data.resize(static_cast<size_t>(h) * w);
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c) a(r, c) = static_cast<float>(0.8 + 0.02 * c);
  return a;
}

static Array2f apply(const Array2f& x, double a, double b) {
  Array2f y = x;
  for (auto& v : y.data) v = static_cast<float>(a * v + b);
  return y;
}

static Array2f ones(int h, int w) {
  Array2f m;
  m.rows = h;
  m.cols = w;
  m.data.assign(static_cast<size_t>(h) * w, 1.f);
  return m;
}

int main() {
  constexpr double kA = -0.06610793956568871;
  constexpr double kB = 1.555980697834118;
  FitConfig cfg;
  cfg.erode_r = 0;
  cfg.min_gt_std = 0.0;
  auto pred = ramp(32, 48);
  auto gt = apply(pred, kA, kB);
  auto mask = ones(32, 48);
  auto m = spur::moments_from_image(pred, gt, mask, cfg);
  auto r = spur::solve(m);
  if (std::abs(r.alpha - kA) > 1e-6 || std::abs(r.beta - kB) > 1e-6) {
    std::cerr << "ramp mismatch alpha=" << r.alpha << " beta=" << r.beta << "\n";
    return 1;
  }
  try {
    spur::solve(spur::Moments{});
    std::cerr << "empty should throw\n";
    return 1;
  } catch (const std::runtime_error&) {
  }

  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "spur_scale_shift_smoke";
  fs::create_directories(tmp);
  std::vector<spur::PairPaths> pairs;
  for (int i = 0; i < 8; ++i) {
    auto p = ramp(16, 24);
    for (auto& v : p.data) v += 0.001f * static_cast<float>(i);
    auto g = apply(p, kA, kB);
    auto mk = ones(16, 24);
    auto pp = tmp / ("pred_" + std::to_string(i) + ".npy");
    auto gp = tmp / ("gt_" + std::to_string(i) + ".npy");
    auto mp = tmp / ("mask_" + std::to_string(i) + ".npy");
    spur::write_npy_f32(pp, p);
    spur::write_npy_f32(gp, g);
    spur::write_npy_f32(mp, mk);
    pairs.push_back({pp, gp, mp});
  }
  auto r1 = spur::fit_files(pairs, cfg, 1);
  auto r8 = spur::fit_files(pairs, cfg, 8);
  if (std::memcmp(&r1.alpha, &r8.alpha, sizeof(double)) != 0 ||
      std::memcmp(&r1.beta, &r8.beta, sizeof(double)) != 0) {
    std::cerr << "1-thread vs 8-thread not bit-identical\n";
    return 1;
  }
  std::cout << "smoke ok  alpha=" << r1.alpha << "  beta=" << r1.beta << "\n";
  return 0;
}
