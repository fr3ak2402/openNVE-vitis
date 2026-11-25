// C++ bridge between Rust (cxx) and the Vitis AI Runtime (VART, XIR).
//
// A compiled model (.xmodel) holds one DPU subgraph (the network on the DPU) and subgraphs for the
// host (input, sometimes a softmax at the end). The bridge runs the DPU subgraph: it quantizes
// float32 inputs to the int8 values of the DPU (scale 2^fix_point), runs the DPU and converts the
// int8 outputs back to float32. Tensors are channels last (NHWC).
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rust/cxx.h"

namespace xir {
class Graph;
class Subgraph;
class Attrs;
}  // namespace xir

namespace vart {
class RunnerExt;
class TensorBuffer;
}  // namespace vart

namespace vitis_runtime {

struct TensorInfo;
struct ModelInfo;
struct CoreInfo;

class Model {
public:
    Model(std::unique_ptr<xir::Graph> graph, const xir::Subgraph* subgraph, std::unique_ptr<xir::Attrs> attrs,
          std::unique_ptr<vart::RunnerExt> runner);
    ~Model();

    rust::Vec<TensorInfo> inputs() const;
    rust::Vec<TensorInfo> outputs() const;
    // frames the DPU runs at once (the batch of the DPU subgraph)
    uint32_t batch_size() const;
    // input: count frames, frame after frame; output: for each output (in the order of outputs())
    // count frames, frame after frame
    void run(rust::Slice<const float> input, size_t count, rust::Slice<float> output);

private:
    // destroyed in reverse order: the runner first, the graph last
    std::unique_ptr<xir::Graph> m_graph;
    const xir::Subgraph* m_subgraph;
    std::unique_ptr<xir::Attrs> m_attrs;
    std::unique_ptr<vart::RunnerExt> m_runner;
    std::vector<vart::TensorBuffer*> m_inputs;
    std::vector<vart::TensorBuffer*> m_outputs;
    uint32_t m_batch;
};

rust::String library_version();
// structure of a model without a DPU (for checks and for the dry run of the Python package)
ModelInfo inspect_model(rust::Str path);
// runner: "" (the DPU) or a VART runner library, for instance "libvart-dummy-runner.so" (tests)
std::unique_ptr<Model> load_model(rust::Str path, rust::Str runner);
// DPU cores of the loaded FPGA design (name, fingerprint). Aborts the process if the device can not
// be opened (VART checks with glog CHECK): call it only in a separate process.
rust::Vec<CoreInfo> dpu_cores();
// writes a small .xmodel with one DPU subgraph (for tests without a compiled model)
void write_test_model(rust::Str path, uint32_t height, uint32_t width, uint32_t channels_in, uint32_t channels_out,
                      int32_t fix_in, int32_t fix_out);

}  // namespace vitis_runtime
