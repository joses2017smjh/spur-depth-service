#include "spur/fit_scale_shift.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<spur::PairPaths> read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<spur::PairPaths> out;
  std::string line;
  int i = 0;
  while (std::getline(in, line)) {
    ++i;
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    std::string a, b, c;
    if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',') || !std::getline(ss, c, ',')) {
      throw std::runtime_error(path + ":" + std::to_string(i) + ": expected pred,gt,mask");
    }
    if (i == 1 && (a == "pred" || a == "pro" || a == "x")) continue;
    // trim trailing CR
    if (!c.empty() && c.back() == '\r') c.pop_back();
    out.push_back({a, b, c});
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string csv;
  int threads = 1;
  spur::FitConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
      return argv[++i];
    };
    if (a == "--pairs-csv") csv = need("--pairs-csv");
    else if (a == "--threads") threads = std::stoi(need("--threads"));
    else if (a == "--erode-r") cfg.erode_r = std::stoi(need("--erode-r"));
    else if (a == "--min-gt-std") cfg.min_gt_std = std::stod(need("--min-gt-std"));
    else if (a == "--min-depth") cfg.min_depth = std::stod(need("--min-depth"));
    else if (a == "--max-depth") cfg.max_depth = std::stod(need("--max-depth"));
    else if (a == "--help" || a == "-h") {
      std::cout << "scale_shift_fit --pairs-csv FILE [--threads N] [--erode-r R]\n";
      return 0;
    } else {
      std::cerr << "unknown flag " << a << "\n";
      return 2;
    }
  }
  if (csv.empty()) {
    std::cerr << "Data/full_spur is not mounted and --pairs-csv was not given.\n";
    return 2;
  }
  try {
    auto pairs = read_csv(csv);
    auto r = spur::fit_files(pairs, cfg, threads);
    std::cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
    std::cout.precision(16);
    std::cout << "alpha=" << r.alpha << "  beta=" << r.beta << "  n_images=" << r.n_images_used
              << "  n_pixels=" << r.moments.n << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 2;
  }
}
