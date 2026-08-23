#pragma once

#include "spur/npy.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace spur {

struct Moments {
  double n = 0;
  double sx = 0;
  double sy = 0;
  double sxy = 0;
  double sx2 = 0;

  Moments& merge(const Moments& o) {
    n += o.n;
    sx += o.sx;
    sy += o.sy;
    sxy += o.sxy;
    sx2 += o.sx2;
    return *this;
  }
};

struct FitConfig {
  int erode_r = 10;
  double min_gt_std = 0.05;
  double min_depth = 0.5;
  double max_depth = 10.0;
};

struct PairPaths {
  std::filesystem::path pred;
  std::filesystem::path gt;
  std::filesystem::path mask;
};

struct FitResult {
  double alpha = 0;
  double beta = 0;
  Moments moments;
  int n_images_used = 0;
};

Moments moments_from_image(const Array2f& pred, const Array2f& gt, const Array2f& mask,
                           const FitConfig& cfg);

FitResult solve(const Moments& m);

// Parallelise over files; merge Moments in file-index order so 1-thread and
// N-thread results are bit-identical.
FitResult fit_files(const std::vector<PairPaths>& pairs, const FitConfig& cfg, int threads);

}  // namespace spur
