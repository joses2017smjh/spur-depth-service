// C2 TensorRT harness: deserialize a .plan and print I/O names + bytes.
// Does not invent a latency number. Python `spur_depth.export.trt_runtime`
// is the timed path (torch CUDA bindings). This binary proves the engine
// file is a real TensorRT blob, not a renamed ONNX.

#include <NvInfer.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

class Log : public nvinfer1::ILogger {
 public:
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: trt_harness ENGINE.plan\n";
    return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  std::vector<char> blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (blob.size() < 8) {
    std::cerr << "file too small to be a TensorRT engine\n";
    return 2;
  }
  Log logger;
  auto runtime = nvinfer1::createInferRuntime(logger);
  if (!runtime) {
    std::cerr << "createInferRuntime failed\n";
    return 2;
  }
  auto engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!engine) {
    std::cerr << "deserializeCudaEngine failed (wrong TRT version or corrupt plan)\n";
    return 2;
  }
  std::cout << "plan=" << argv[1] << " bytes=" << blob.size()
            << " io=" << engine->getNbIOTensors() << "\n";
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    auto name = engine->getIOTensorName(i);
    auto mode = engine->getTensorIOMode(name);
    std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "in " : "out ") << name << "\n";
  }
  delete engine;
  delete runtime;
  return 0;
}
