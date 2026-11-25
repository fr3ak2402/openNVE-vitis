#include "vitis-runtime/cpp/shim.hpp"
#include "vitis-runtime/src/lib.rs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/dpu_controller.hpp>
#include <dlfcn.h>
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>
#include <xir/op/op.hpp>
#include <xir/tensor/tensor.hpp>

#ifndef OPENNVE_VART_VERSION
#define OPENNVE_VART_VERSION "unknown"
#endif

namespace vitis_runtime {

namespace {

std::string device_of(const xir::Subgraph* subgraph) {
    return subgraph->has_attr("device") ? subgraph->get_attr<std::string>("device") : std::string();
}

std::vector<const xir::Subgraph*> children(const xir::Graph* graph) {
    auto root = graph->get_root_subgraph();
    if (root->is_leaf()) {
        return {};
    }
    return root->children_topological_sort();
}

const xir::Subgraph* dpu_subgraph(const xir::Graph* graph, const std::string& path) {
    std::vector<const xir::Subgraph*> found;
    for (auto child : children(graph)) {
        if (device_of(child) == "DPU") {
            found.push_back(child);
        }
    }
    if (found.empty()) {
        throw std::runtime_error(path + " has no DPU subgraph: compile the model with the Vitis AI compiler "
                                        "for the DPU of the board (vai_c_xir / vai_c_tensorflow2 with the arch.json of the DPU)");
    }
    if (found.size() > 1) {
        throw std::runtime_error(
            path + " has " + std::to_string(found.size()) +
            " DPU subgraphs: the network is split by layers the DPU does not support (they run on the CPU). "
            "openNVE-vitis runs models with one DPU subgraph: replace or remove those layers "
            "(a final softmax / sigmoid is not needed, openNVE applies the activation of \"prediction\")");
    }
    return found.front();
}

int fix_point_of(const xir::Tensor* tensor) {
    if (!tensor->has_attr("fix_point")) {
        throw std::runtime_error("tensor " + tensor->get_name() + " has no fix_point (not quantized)");
    }
    return tensor->get_attr<int>("fix_point");
}

TensorInfo tensor_info(const xir::Tensor* tensor) {
    TensorInfo info;
    info.name = tensor->get_name();
    auto shape = tensor->get_shape();
    size_t size = 1;
    for (size_t i = 1; i < shape.size(); ++i) {  // without the batch
        info.shape.push_back(static_cast<uint32_t>(shape[i]));
        size *= static_cast<size_t>(shape[i]);
    }
    info.frame_size = size;
    info.fix_point = tensor->has_attr("fix_point") ? tensor->get_attr<int>("fix_point") : 0;
    auto type = tensor->get_data_type();
    info.data_type = type.to_string();
    info.bits = static_cast<uint32_t>(type.bit_width);
    return info;
}

void check_int8(const xir::Tensor* tensor) {
    auto type = tensor->get_data_type();
    if ((type.type != xir::DataType::XINT && type.type != xir::DataType::XUINT && type.type != xir::DataType::INT &&
         type.type != xir::DataType::UINT) ||
        type.bit_width != 8) {
        throw std::runtime_error("tensor " + tensor->get_name() + " has the data type " + type.to_string() +
                                 ": only 8 bit DPU tensors are supported");
    }
    fix_point_of(tensor);
}

bool is_signed(const xir::Tensor* tensor) {
    auto type = tensor->get_data_type().type;
    return type == xir::DataType::XINT || type == xir::DataType::INT;
}

std::unique_ptr<xir::Graph> read_graph(const std::string& path) {
    try {
        return xir::Graph::deserialize(path);
    } catch (const std::exception& error) {
        throw std::runtime_error("can not read " + path + ": " + error.what());
    }
}

}  // namespace

// ------------------------------------------------------------------------------------------- Model
Model::Model(std::unique_ptr<xir::Graph> graph, const xir::Subgraph* subgraph, std::unique_ptr<xir::Attrs> attrs,
             std::unique_ptr<vart::RunnerExt> runner)
    : m_graph(std::move(graph)), m_subgraph(subgraph), m_attrs(std::move(attrs)), m_runner(std::move(runner)) {
    m_inputs = m_runner->get_inputs();
    m_outputs = m_runner->get_outputs();
    if (m_inputs.size() != 1) {
        throw std::runtime_error("the DPU subgraph has " + std::to_string(m_inputs.size()) +
                                 " inputs: openNVE-vitis runs models with one input");
    }
    if (m_outputs.empty()) {
        throw std::runtime_error("the DPU subgraph has no output");
    }
    for (auto buffer : m_inputs) check_int8(buffer->get_tensor());
    for (auto buffer : m_outputs) check_int8(buffer->get_tensor());
    auto shape = m_inputs.front()->get_tensor()->get_shape();
    m_batch = shape.empty() ? 1u : static_cast<uint32_t>(std::max(1, shape.front()));
}

Model::~Model() {
    m_runner.reset();
    m_attrs.reset();
    m_graph.reset();
}

rust::Vec<TensorInfo> Model::inputs() const {
    rust::Vec<TensorInfo> result;
    for (auto buffer : m_inputs) result.push_back(tensor_info(buffer->get_tensor()));
    return result;
}

rust::Vec<TensorInfo> Model::outputs() const {
    rust::Vec<TensorInfo> result;
    for (auto buffer : m_outputs) result.push_back(tensor_info(buffer->get_tensor()));
    return result;
}

uint32_t Model::batch_size() const { return m_batch; }

namespace {

size_t frame_bytes(const vart::TensorBuffer* buffer) {
    auto tensor = buffer->get_tensor();
    auto shape = tensor->get_shape();
    return static_cast<size_t>(tensor->get_data_size()) / static_cast<size_t>(std::max(1, shape.front()));
}

uint64_t frame_address(vart::TensorBuffer* buffer, int index, size_t needed) {
    auto rank = buffer->get_tensor()->get_shape().size();
    std::vector<int32_t> position(rank, 0);
    position[0] = index;
    uint64_t address = 0;
    size_t size = 0;
    std::tie(address, size) = buffer->data(position);
    if (address == 0 || size < needed) {
        throw std::runtime_error("tensor buffer of " + buffer->get_tensor()->get_name() + " is too small");
    }
    return address;
}

void quantize(const float* in, size_t count, float scale, bool signed_values, void* out) {
    const float low = signed_values ? -128.0f : 0.0f;
    const float high = signed_values ? 127.0f : 255.0f;
    if (signed_values) {
        auto target = static_cast<int8_t*>(out);
        for (size_t i = 0; i < count; ++i) {
            target[i] = static_cast<int8_t>(std::clamp(std::nearbyint(in[i] * scale), low, high));
        }
    } else {
        auto target = static_cast<uint8_t*>(out);
        for (size_t i = 0; i < count; ++i) {
            target[i] = static_cast<uint8_t>(std::clamp(std::nearbyint(in[i] * scale), low, high));
        }
    }
}

void dequantize(const void* in, size_t count, float scale, bool signed_values, float* out) {
    if (signed_values) {
        auto source = static_cast<const int8_t*>(in);
        for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(source[i]) * scale;
    } else {
        auto source = static_cast<const uint8_t*>(in);
        for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(source[i]) * scale;
    }
}

}  // namespace

void Model::run(rust::Slice<const float> input, size_t count, rust::Slice<float> output) {
    auto in_buffer = m_inputs.front();
    auto in_tensor = in_buffer->get_tensor();
    auto in_info = tensor_info(in_tensor);
    if (input.size() != count * in_info.frame_size) {
        throw std::runtime_error("input has " + std::to_string(input.size()) + " values, expected " +
                                 std::to_string(count * in_info.frame_size));
    }
    std::vector<TensorInfo> out_infos;
    size_t total = 0;
    for (auto buffer : m_outputs) {
        out_infos.push_back(tensor_info(buffer->get_tensor()));
        total += out_infos.back().frame_size * count;
    }
    if (output.size() != total) {
        throw std::runtime_error("output has " + std::to_string(output.size()) + " values, expected " +
                                 std::to_string(total));
    }
    const float in_scale = std::ldexp(1.0f, fix_point_of(in_tensor));
    const bool in_signed = is_signed(in_tensor);
    const size_t in_bytes = frame_bytes(in_buffer);
    if (in_bytes < in_info.frame_size) {
        throw std::runtime_error("the DPU input buffer is smaller than a frame");
    }

    for (size_t start = 0; start < count; start += m_batch) {
        const size_t frames = std::min<size_t>(m_batch, count - start);
        for (size_t b = 0; b < frames; ++b) {
            auto address = frame_address(in_buffer, static_cast<int>(b), in_info.frame_size);
            quantize(input.data() + (start + b) * in_info.frame_size, in_info.frame_size, in_scale, in_signed,
                     reinterpret_cast<void*>(address));
        }
        for (auto buffer : m_inputs) buffer->sync_for_write(0, frame_bytes(buffer));
        auto job = m_runner->execute_async(m_inputs, m_outputs);
        if (job.second != 0) {
            throw std::runtime_error("the DPU did not accept the job (status " + std::to_string(job.second) + ")");
        }
        auto status = m_runner->wait(static_cast<int>(job.first), -1);
        if (status != 0) {
            throw std::runtime_error("the DPU job failed (status " + std::to_string(status) + ")");
        }
        for (auto buffer : m_outputs) buffer->sync_for_read(0, frame_bytes(buffer));

        size_t offset = 0;
        for (size_t o = 0; o < m_outputs.size(); ++o) {
            auto buffer = m_outputs[o];
            auto tensor = buffer->get_tensor();
            const auto& info = out_infos[o];
            const float scale = std::ldexp(1.0f, -fix_point_of(tensor));
            const bool out_signed = is_signed(tensor);
            for (size_t b = 0; b < frames; ++b) {
                auto address = frame_address(buffer, static_cast<int>(b), info.frame_size);
                dequantize(reinterpret_cast<const void*>(address), info.frame_size, scale, out_signed,
                           output.data() + offset + (start + b) * info.frame_size);
            }
            offset += info.frame_size * count;
        }
    }
}

// ------------------------------------------------------------------------------------------- free functions
rust::String library_version() { return rust::String(OPENNVE_VART_VERSION); }

ModelInfo inspect_model(rust::Str path_str) {
    std::string path(path_str);
    auto graph = read_graph(path);
    ModelInfo info;
    info.batch = 0;
    info.fingerprint = 0;
    info.dpu_subgraphs = 0;
    for (auto child : children(graph.get())) {
        auto device = device_of(child);
        info.devices.push_back(rust::String(device));
        if (device != "DPU") continue;
        info.dpu_subgraphs += 1;
        if (info.dpu_subgraphs > 1) continue;
        for (auto tensor : child->get_sorted_input_tensors()) info.inputs.push_back(tensor_info(tensor));
        for (auto tensor : child->get_sorted_output_tensors()) info.outputs.push_back(tensor_info(tensor));
        auto inputs = child->get_sorted_input_tensors();
        if (!inputs.empty() && !inputs.front()->get_shape().empty()) {
            info.batch = static_cast<uint32_t>(inputs.front()->get_shape().front());
        }
        if (child->has_attr("dpu_name")) info.target = child->get_attr<std::string>("dpu_name");
        if (child->has_attr("dpu_fingerprint")) info.fingerprint = child->get_attr<uint64_t>("dpu_fingerprint");
    }
    return info;
}

std::unique_ptr<Model> load_model(rust::Str path_str, rust::Str runner_str) {
    std::string path(path_str);
    std::string runner_lib(runner_str);
    auto graph = read_graph(path);
    auto subgraph = dpu_subgraph(graph.get(), path);
    for (auto tensor : subgraph->get_sorted_input_tensors()) check_int8(tensor);
    for (auto tensor : subgraph->get_sorted_output_tensors()) check_int8(tensor);
    auto attrs = xir::Attrs::create();
    if (!runner_lib.empty()) {
        attrs->set_attr("lib", std::map<std::string, std::string>{{"DPU", runner_lib}});
    }
    std::unique_ptr<vart::RunnerExt> runner;
    try {
        runner = vart::RunnerExt::create_runner(subgraph, attrs.get());
    } catch (const std::exception& error) {
        throw std::runtime_error("can not create the DPU runner for " + path + ": " + error.what());
    }
    if (!runner) {
        throw std::runtime_error("can not create the DPU runner for " + path);
    }
    return std::make_unique<Model>(std::move(graph), subgraph, std::move(attrs), std::move(runner));
}

rust::Vec<CoreInfo> dpu_cores() {
    // libvart-dpu-controller needs XRT: loaded only here, so that the module works without XRT
    auto library = dlopen("libvart-dpu-controller.so.5", RTLD_NOW | RTLD_GLOBAL);
    if (library == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error(std::string("can not load the DPU controller (is XRT installed?): ") +
                                 (error ? error : "unknown error"));
    }
    using Exists = bool (*)();
    using Instance = std::shared_ptr<xir::DpuController> (*)();
    auto exists = reinterpret_cast<Exists>(dlsym(library, "_ZN3xir13DpuController17exist_dpu_devicesEv"));
    auto instance = reinterpret_cast<Instance>(dlsym(library, "_ZN3xir13DpuController12get_instanceEv"));
    if (exists == nullptr || instance == nullptr) {
        throw std::runtime_error("the DPU controller library has no xir::DpuController");
    }
    rust::Vec<CoreInfo> result;
    if (!exists()) {
        return result;
    }
    auto controller = instance();
    for (size_t i = 0; i < controller->get_num_of_dpus(); ++i) {
        CoreInfo core;
        core.name = controller->get_full_name(i);
        core.fingerprint = controller->get_fingerprint(i);
        core.batch = static_cast<uint32_t>(controller->get_batch_size(i));
        result.push_back(core);
    }
    return result;
}

void write_test_model(rust::Str path_str, uint32_t height, uint32_t width, uint32_t channels_in,
                      uint32_t /*channels_out*/, int32_t fix_in, int32_t fix_out) {
    std::string path(path_str);
    auto graph = xir::Graph::create("opennve_vitis_test");
    auto input_attrs = xir::Attrs::create();
    input_attrs->set_attr("shape", std::vector<int32_t>{1, static_cast<int32_t>(height), static_cast<int32_t>(width),
                                                       static_cast<int32_t>(channels_in)});
    input_attrs->set_attr("data_type", std::string("XINT8"));
    auto input = graph->add_op("input", "data-fix", std::move(input_attrs), {});
    input->get_output_tensor()->set_attr("fix_point", fix_in);
    auto output = graph->add_op("output", "relu", xir::Attrs::create(), {{"input", {input}}});
    output->get_output_tensor()->set_attr("fix_point", fix_out);

    auto root = graph->get_root_subgraph();
    root->create_children();
    for (auto child : root->get_children()) {
        if (child->has_op(input)) {
            child->set_attr("device", std::string("USER"));
        } else {
            child->set_attr("device", std::string("DPU"));
            child->set_attr("dpu_name", std::string("DPUCZDX8G_ISA1_B4096"));
            child->set_attr("dpu_fingerprint", static_cast<uint64_t>(0x101000016010407ULL));
            child->set_attr("runner", std::map<std::string, std::string>{{"run", "libvart-dpu-runner.so"},
                                                                          {"ref", "libvart-cpu-runner.so"}});
        }
    }
    graph->serialize(path);
}

}  // namespace vitis_runtime
