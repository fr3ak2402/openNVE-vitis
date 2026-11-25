//! Rust bindings for the Vitis AI Runtime (VART): run compiled models (`.xmodel`) on the DPU of
//! AMD Kria boards (KV260, KR260, K26 / K24 SOMs) and other Zynq UltraScale+ MPSoCs.
//!
//! A `.xmodel` holds the quantized network for one DPU configuration (for instance
//! DPUCZDX8G_ISA1_B4096 of the KV260). The model's DPU subgraph runs on the DPU; inputs and outputs
//! are `f32`, channels last (height × width × channels). The crate converts them to and from the
//! 8 bit values of the DPU (scale 2^fix_point of each tensor).
//!
//! ```no_run
//! let model = vitis_runtime::Model::load("unet.xmodel")?;
//! let input = vec![0.0f32; 4 * model.input().frame_size];
//! let outputs = model.run(&input, 4)?; // one Vec<f32> per output, 4 frames each
//! # Ok::<(), vitis_runtime::Error>(())
//! ```
//!
//! VART stops the process (glog CHECK) on some errors, for instance if no DPU is loaded. Check the
//! board first ([`dpu_cores`] in a separate process does that) before loading models.

use std::sync::Mutex;

#[cxx::bridge(namespace = "vitis_runtime")]
mod ffi {
    /// An input or output of the DPU subgraph: name, shape without the batch (height × width ×
    /// channels), values per frame, fix point (value = integer × 2^-fix_point), data type.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TensorInfo {
        name: String,
        shape: Vec<u32>,
        frame_size: usize,
        fix_point: i32,
        data_type: String,
        bits: u32,
    }

    /// The structure of a .xmodel (read without a DPU).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ModelInfo {
        /// inputs and outputs of the (first) DPU subgraph
        inputs: Vec<TensorInfo>,
        outputs: Vec<TensorInfo>,
        /// batch of the DPU subgraph
        batch: u32,
        /// number of DPU subgraphs (openNVE-vitis runs models with one)
        dpu_subgraphs: u32,
        /// device of every subgraph in order: "USER", "DPU", "CPU"
        devices: Vec<String>,
        /// DPU the model was compiled for, for instance DPUCZDX8G_ISA1_B4096
        target: String,
        fingerprint: u64,
    }

    /// A DPU core of the loaded FPGA design.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CoreInfo {
        name: String,
        fingerprint: u64,
        batch: u32,
    }

    unsafe extern "C++" {
        include!("vitis-runtime/cpp/shim.hpp");

        type Model;

        fn library_version() -> String;
        fn inspect_model(path: &str) -> Result<ModelInfo>;
        fn load_model(path: &str, runner: &str) -> Result<UniquePtr<Model>>;
        fn dpu_cores() -> Result<Vec<CoreInfo>>;
        fn write_test_model(
            path: &str,
            height: u32,
            width: u32,
            channels_in: u32,
            channels_out: u32,
            fix_in: i32,
            fix_out: i32,
        ) -> Result<()>;

        fn inputs(self: &Model) -> Vec<TensorInfo>;
        fn outputs(self: &Model) -> Vec<TensorInfo>;
        fn batch_size(self: &Model) -> u32;
        fn run(
            self: Pin<&mut Model>,
            input: &[f32],
            count: usize,
            output: &mut [f32],
        ) -> Result<()>;
    }
}

pub use ffi::{CoreInfo, ModelInfo, TensorInfo};

/// An error of the runtime (or of the model).
#[derive(Debug, Clone)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<cxx::Exception> for Error {
    fn from(exception: cxx::Exception) -> Self {
        Error(exception.what().to_owned())
    }
}

/// Version of the Vitis AI Runtime this crate was built with, for instance "5.0.0".
pub fn library_version() -> String {
    ffi::library_version()
}

/// Reads the structure of a .xmodel (no DPU needed).
pub fn inspect(path: impl AsRef<std::path::Path>) -> Result<ModelInfo, Error> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(Error(format!("{} not found", path.display())));
    }
    Ok(ffi::inspect_model(&path.to_string_lossy())?)
}

/// The DPU cores of the FPGA design that is loaded (empty without DPU).
///
/// VART stops the process if the device can not be opened: call this in a separate process.
pub fn dpu_cores() -> Result<Vec<CoreInfo>, Error> {
    Ok(ffi::dpu_cores()?)
}

/// Writes a small .xmodel with one DPU subgraph (a ReLU) for tests without a compiled model. It
/// does not run on a real DPU (it has no DPU code); run it with the runner
/// "libvart-dummy-runner.so".
#[doc(hidden)]
pub fn write_test_model(
    path: impl AsRef<std::path::Path>,
    height: u32,
    width: u32,
    channels: u32,
    fix_in: i32,
    fix_out: i32,
) -> Result<(), Error> {
    Ok(ffi::write_test_model(
        &path.as_ref().to_string_lossy(),
        height,
        width,
        channels,
        channels,
        fix_in,
        fix_out,
    )?)
}

/// A compiled model (.xmodel) on the DPU.
///
/// `run` can be called from several threads; calls on one model run one after the other (use one
/// model per thread to keep several DPU cores busy).
pub struct Model {
    inner: Mutex<cxx::UniquePtr<ffi::Model>>,
    input: TensorInfo,
    outputs: Vec<TensorInfo>,
    batch_size: u32,
}

// The C++ model is only used under the mutex; VART runners can be used from any thread.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Loads a .xmodel and creates a DPU runner for its DPU subgraph.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        Self::load_with_runner(path, "")
    }

    /// Like `load`, with another VART runner library for the DPU subgraph ("" = the DPU), for
    /// instance "libvart-dummy-runner.so" (runs nothing, for tests without a DPU).
    pub fn load_with_runner(
        path: impl AsRef<std::path::Path>,
        runner: &str,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(Error(format!("{} not found", path.display())));
        }
        let inner = ffi::load_model(&path.to_string_lossy(), runner)?;
        let input = inner
            .inputs()
            .into_iter()
            .next()
            .ok_or_else(|| Error("the model has no input".into()))?;
        let outputs = inner.outputs();
        let batch_size = inner.batch_size();
        Ok(Model {
            inner: Mutex::new(inner),
            input,
            outputs,
            batch_size,
        })
    }

    pub fn input(&self) -> &TensorInfo {
        &self.input
    }

    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// Frames the DPU runs at once.
    pub fn batch_size(&self) -> u32 {
        self.batch_size
    }

    /// Runs `count` frames (`input`: frame after frame, `count * input().frame_size` values) and
    /// writes the outputs into `output`: for each output (in the order of `outputs()`) `count`
    /// frames.
    pub fn run_into(&self, input: &[f32], count: usize, output: &mut [f32]) -> Result<(), Error> {
        let mut inner = self.inner.lock().unwrap();
        let model = inner
            .as_mut()
            .ok_or_else(|| Error("the model is closed".into()))?;
        Ok(model.run(input, count, output)?)
    }

    /// Runs `count` frames and returns one vector per output.
    pub fn run(&self, input: &[f32], count: usize) -> Result<Vec<Vec<f32>>, Error> {
        let total: usize = self
            .outputs
            .iter()
            .map(|output| output.frame_size * count)
            .sum();
        let mut output = vec![0.0f32; total];
        self.run_into(input, count, &mut output)?;
        let mut result = Vec::with_capacity(self.outputs.len());
        let mut rest = output.as_slice();
        for info in &self.outputs {
            let (head, tail) = rest.split_at(info.frame_size * count);
            result.push(head.to_vec());
            rest = tail;
        }
        Ok(result)
    }

    /// Releases the runner (also done when the model is dropped).
    pub fn close(&self) {
        *self.inner.lock().unwrap() = cxx::UniquePtr::null();
    }
}
