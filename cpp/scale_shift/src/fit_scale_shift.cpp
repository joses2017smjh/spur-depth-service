#include "spur/fit_scale_shift.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spur {
namespace {

Array2f erode(const Array2f& mask, int radius) {
  Array2f out;
  out.rows = mask.rows;
  out.cols = mask.cols;
  out.data.assign(mask.data.size(), 0.f);
  if (radius <= 0) {
    out.data = mask.data;
    for (auto& v : out.data) v = v > 0.f ? 1.f : 0.f;
    return out;
  }
  for (int r = 0; r < mask.rows; ++r) {
    for (int c = 0; c < mask.cols; ++c) {
      bool ok = true;
      for (int dr = -radius; dr <= radius && ok; ++dr) {
        for (int dc = -radius; dc <= radius; ++dc) {
          int rr = r + dr;
          int cc = c + dc;
          if (rr < 0 || cc < 0 || rr >= mask.rows || cc >= mask.cols || mask(rr, cc) <= 0.f) {
            ok = false;
            break;
          }
        }
      }
      out(r, c) = ok ? 1.f : 0.f;
    }
  }
  return out;
}

}  // namespace

Moments moments_from_image(const Array2f& pred, const Array2f& gt, const Array2f& mask,
                           const FitConfig& cfg) {
  if (pred.rows != gt.rows || pred.cols != gt.cols || mask.rows != pred.rows ||
      mask.cols != pred.cols) {
    throw std::runtime_error("pred/gt/mask shape mismatch");
  }
  Array2f eroded = erode(mask, cfg.erode_r);
  std::vector<double> xs;
  std::vector<double> ys;
  xs.reserve(pred.data.size());
  ys.reserve(pred.data.size());
  for (int r = 0; r < pred.rows; ++r) {
    for (int c = 0; c < pred.cols; ++c) {
      if (eroded(r, c) <= 0.f) continue;
      double x = pred(r, c);
      double y = gt(r, c);
      if (!std::isfinite(x) || !std::isfinite(y)) continue;
      if (y < cfg.min_depth || y > cfg.max_depth) continue;
      xs.push_back(x);
      ys.push_back(y);
    }
  }
  if (xs.size() < 2) return Moments{};
  double mean = 0;
  for (double y : ys) mean += y;
  mean /= static_cast<double>(ys.size());
  double var = 0;
  for (double y : ys) {
    double d = y - mean;
    var += d * d;
  }
  double stdev = std::sqrt(var / static_cast<double>(ys.size()));
  if (stdev < cfg.min_gt_std) return Moments{};

  Moments m;
  m.n = static_cast<double>(xs.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    m.sx += xs[i];
    m.sy += ys[i];
    m.sxy += xs[i] * ys[i];
    m.sx2 += xs[i] * xs[i];
  }
  return m;
}

FitResult solve(const Moments& m) {
  if (m.n < 2) throw std::runtime_error("no valid pixels: empty or fully-masked input");
  double det = m.sx2 * m.n - m.sx * m.sx;
  if (std::abs(det) < 1e-18) throw std::runtime_error("singular normal equations");
  FitResult r;
  r.alpha = (m.sxy * m.n - m.sx * m.sy) / det;
  r.beta = (m.sx2 * m.sy - m.sx * m.sxy) / det;
  r.moments = m;
  return r;
}

FitResult fit_files(const std::vector<PairPaths>& pairs, const FitConfig& cfg, int threads) {
  const int n = static_cast<int>(pairs.size());
  std::vector<Moments> per_file(static_cast<size_t>(n));
  std::vector<std::string> errors(static_cast<size_t>(n));

#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    try {
      Array2f pred = read_npy_f32(pairs[static_cast<size_t>(i)].pred);
      Array2f gt = read_npy_f32(pairs[static_cast<size_t>(i)].gt);
      Array2f mask = read_npy_f32(pairs[static_cast<size_t>(i)].mask);
      per_file[static_cast<size_t>(i)] = moments_from_image(pred, gt, mask, cfg);
    } catch (const std::exception& ex) {
      errors[static_cast<size_t>(i)] = ex.what();
    }
  }
#else
  (void)threads;
  for (int i = 0; i < n; ++i) {
    Array2f pred = read_npy_f32(pairs[static_cast<size_t>(i)].pred);
    Array2f gt = read_npy_f32(pairs[static_cast<size_t>(i)].gt);
    Array2f mask = read_npy_f32(pairs[static_cast<size_t>(i)].mask);
    per_file[static_cast<size_t>(i)] = moments_from_image(pred, gt, mask, cfg);
  }
#endif

  for (const auto& e : errors) {
    if (!e.empty()) throw std::runtime_error(e);
  }

  Moments acc;
  int used = 0;
  for (const auto& m : per_file) {
    if (m.n < 2) continue;
    acc.merge(m);
    ++used;
  }
  FitResult r = solve(acc);
  r.n_images_used = used;
  return r;
}

}  // namespace spur
