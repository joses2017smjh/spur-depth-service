#include "spur/fit_scale_shift.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr double kAlpha = -0.06610793956568871;
constexpr double kBeta = 1.555980697834118;

spur::Array2f ramp(int h, int w) {
  spur::Array2f a;
  a.rows = h;
  a.cols = w;
  a.data.resize(static_cast<size_t>(h) * w);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      a(r, c) = static_cast<float>(0.8 + 0.02 * c);
    }
  }
  return a;
}

spur::Array2f apply_affine(const spur::Array2f& x, double a, double b) {
  spur::Array2f y = x;
  for (auto& v : y.data) v = static_cast<float>(a * v + b);
  return y;
}

spur::Array2f ones(int h, int w) {
  spur::Array2f m;
  m.rows = h;
  m.cols = w;
  m.data.assign(static_cast<size_t>(h) * w, 1.f);
  return m;
}

spur::Array2f zeros(int h, int w) {
  spur::Array2f m;
  m.rows = h;
  m.cols = w;
  m.data.assign(static_cast<size_t>(h) * w, 0.f);
  return m;
}

}  // namespace

TEST(ScaleShift, KnownAnswerRamp) {
  auto pred = ramp(32, 48);
  auto gt = apply_affine(pred, kAlpha, kBeta);
  auto mask = ones(32, 48);
  spur::FitConfig cfg;
  cfg.erode_r = 0;
  cfg.min_gt_std = 0.0;
  auto m = spur::moments_from_image(pred, gt, mask, cfg);
  auto r = spur::solve(m);
  EXPECT_NEAR(r.alpha, kAlpha, 1e-6);
  EXPECT_NEAR(r.beta, kBeta, 1e-6);
}

TEST(ScaleShift, AllMaskedThrows) {
  auto pred = ramp(8, 8);
  auto gt = apply_affine(pred, kAlpha, kBeta);
  auto mask = zeros(8, 8);
  spur::FitConfig cfg;
  cfg.erode_r = 0;
  cfg.min_gt_std = 0.0;
  auto m = spur::moments_from_image(pred, gt, mask, cfg);
  EXPECT_THROW(spur::solve(m), std::runtime_error);
}

TEST(ScaleShift, EmptyThrows) { EXPECT_THROW(spur::solve(spur::Moments{}), std::runtime_error); }

TEST(ScaleShift, ThreadCountBitIdentical) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "spur_scale_shift_test";
  fs::create_directories(tmp);
  std::vector<spur::PairPaths> pairs;
  for (int i = 0; i < 8; ++i) {
    auto pred = ramp(16, 24);
    for (auto& v : pred.data) v += 0.001f * static_cast<float>(i);
    auto gt = apply_affine(pred, kAlpha, kBeta);
    auto mask = ones(16, 24);
    auto p = tmp / ("pred_" + std::to_string(i) + ".npy");
    auto g = tmp / ("gt_" + std::to_string(i) + ".npy");
    auto m = tmp / ("mask_" + std::to_string(i) + ".npy");
    spur::write_npy_f32(p, pred);
    spur::write_npy_f32(g, gt);
    spur::write_npy_f32(m, mask);
    pairs.push_back({p, g, m});
  }
  spur::FitConfig cfg;
  cfg.erode_r = 0;
  cfg.min_gt_std = 0.0;
  auto r1 = spur::fit_files(pairs, cfg, 1);
  auto r8 = spur::fit_files(pairs, cfg, 8);
  EXPECT_EQ(std::memcmp(&r1.alpha, &r8.alpha, sizeof(double)), 0);
  EXPECT_EQ(std::memcmp(&r1.beta, &r8.beta, sizeof(double)), 0);
  EXPECT_EQ(std::memcmp(&r1.moments.n, &r8.moments.n, sizeof(double)), 0);
  EXPECT_NEAR(r1.alpha, kAlpha, 1e-6);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
