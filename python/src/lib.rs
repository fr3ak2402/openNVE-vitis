//! Python module opennve_vitis._opennve_vitis (the opennve_vitis package wraps it with numpy).

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn error(error: vitis_runtime::Error) -> PyErr {
    PyRuntimeError::new_err(error.0)
}

/// (name, shape without batch, values per frame, fix_point, data type)
type Tensor = (String, Vec<u32>, usize, i32, String);

fn tensor(info: &vitis_runtime::TensorInfo) -> Tensor {
    (
        info.name.clone(),
        info.shape.clone(),
        info.frame_size,
        info.fix_point,
        info.data_type.clone(),
    )
}

/// Version of the bundled Vitis AI Runtime, for instance "5.0.0".
#[pyfunction]
fn library_version() -> String {
    vitis_runtime::library_version()
}

/// Structure of a .xmodel (no DPU needed): dict with inputs, outputs, batch, dpu_subgraphs,
/// devices, target, fingerprint.
#[pyfunction]
fn inspect<'py>(py: Python<'py>, path: std::path::PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let info = py.detach(|| vitis_runtime::inspect(&path)).map_err(error)?;
    let dict = PyDict::new(py);
    dict.set_item("inputs", info.inputs.iter().map(tensor).collect::<Vec<_>>())?;
    dict.set_item(
        "outputs",
        info.outputs.iter().map(tensor).collect::<Vec<_>>(),
    )?;
    dict.set_item("batch", info.batch)?;
    dict.set_item("dpu_subgraphs", info.dpu_subgraphs)?;
    dict.set_item("devices", info.devices)?;
    dict.set_item("target", info.target)?;
    dict.set_item("fingerprint", info.fingerprint)?;
    Ok(dict)
}

/// DPU cores of the loaded FPGA design: list of (name, fingerprint, batch). VART stops the process
/// if the device can not be opened: opennve_vitis calls this in a separate process.
#[pyfunction]
fn dpu_cores(py: Python<'_>) -> PyResult<Vec<(String, u64, u32)>> {
    let cores = py.detach(vitis_runtime::dpu_cores).map_err(error)?;
    Ok(cores
        .into_iter()
        .map(|core| (core.name, core.fingerprint, core.batch))
        .collect())
}

/// Writes a small .xmodel with one DPU subgraph (a ReLU), for tests with the dummy runner.
#[pyfunction]
#[pyo3(signature = (path, height, width, channels, fix_in = 4, fix_out = 3))]
fn write_test_model(
    path: std::path::PathBuf,
    height: u32,
    width: u32,
    channels: u32,
    fix_in: i32,
    fix_out: i32,
) -> PyResult<()> {
    vitis_runtime::write_test_model(&path, height, width, channels, fix_in, fix_out).map_err(error)
}

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}

/// A compiled model on the DPU.
#[pyclass(name = "_Model", module = "opennve_vitis._opennve_vitis", frozen)]
struct Model {
    inner: vitis_runtime::Model,
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (path, runner = ""))]
    fn new(py: Python<'_>, path: std::path::PathBuf, runner: &str) -> PyResult<Self> {
        let runner = runner.to_owned();
        let inner = py
            .detach(|| vitis_runtime::Model::load_with_runner(&path, &runner))
            .map_err(error)?;
        Ok(Model { inner })
    }

    #[getter]
    fn input(&self) -> Tensor {
        tensor(self.inner.input())
    }

    #[getter]
    fn outputs(&self) -> Vec<Tensor> {
        self.inner.outputs().iter().map(tensor).collect()
    }

    #[getter]
    fn batch_size(&self) -> u32 {
        self.inner.batch_size()
    }

    /// Runs count frames: input is a C-contiguous float32 buffer with count frames, output a
    /// writable C-contiguous float32 buffer for all outputs (output after output, count frames
    /// each). Releases the GIL while the DPU runs.
    fn run_into(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        count: usize,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let input = PyBuffer::<f32>::get(input)?;
        let output = PyBuffer::<f32>::get(output)?;
        if !input.is_c_contiguous() || !output.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "input and output must be C-contiguous float32 arrays",
            ));
        }
        if output.readonly() {
            return Err(PyValueError::new_err("output must be writable"));
        }
        let input_slice = SendPtr(input.buf_ptr() as *mut f32);
        let input_len = input.item_count();
        let output_slice = SendPtr(output.buf_ptr() as *mut f32);
        let output_len = output.item_count();
        let inner = &self.inner;
        // the buffers stay alive (input / output hold them) while the GIL is released
        py.detach(move || {
            let input_slice = input_slice;
            let output_slice = output_slice;
            let input =
                unsafe { std::slice::from_raw_parts(input_slice.0 as *const f32, input_len) };
            let output = unsafe { std::slice::from_raw_parts_mut(output_slice.0, output_len) };
            inner.run_into(input, count, output)
        })
        .map_err(error)?;
        drop(input);
        drop(output);
        Ok(())
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| self.inner.close())
    }
}

#[pymodule]
fn _opennve_vitis(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(library_version, module)?)?;
    module.add_function(wrap_pyfunction!(inspect, module)?)?;
    module.add_function(wrap_pyfunction!(dpu_cores, module)?)?;
    module.add_function(wrap_pyfunction!(write_test_model, module)?)?;
    module.add_class::<Model>()?;
    Ok(())
}
