"""DPU of AMD Kria boards for openNVE: run compiled Vitis AI models (.xmodel) with the Vitis AI
Runtime (VART), in Rust.

The package brings its own Vitis AI Runtime; the board needs XRT with the zocl driver and a loaded
DPU design (install.sh sets this up). Inputs and outputs are float32 numpy arrays, channels last
(frames x height x width x channels); the package converts them to and from the 8 bit values of
the DPU.

    import numpy as np, opennve_vitis

    opennve_vitis.status()                       # XRT, zocl, DPU design, DPU cores
    model = opennve_vitis.Model("unet.xmodel")
    model.input                                  # Tensor(name=..., shape=(80, 80, 2), ...)
    outputs = model.run(np.zeros((16, 80, 80, 2), np.float32))   # [array (16, 80, 80, 3)]

VART stops the whole process on some errors (no DPU loaded, a model for another DPU). The package
therefore checks the board and every model once in a separate process before it loads the model
in this one (Model(..., check=False) skips that).

It is also the openNVE plugin "vitis" (opennve_vitis.opennve_plugin): agents with .xmodel models.
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import subprocess
import sys
import threading
import typing

import numpy as np

from . import _opennve_vitis

__version__: str = _opennve_vitis.__version__
VART_VERSION: str = _opennve_vitis.library_version()

__all__ = ["Core", "Model", "ModelInfo", "Tensor", "dpu_cores", "firmware", "inspect", "status"]

VART_CONF = "/etc/vart.conf"
XRT_LIBRARIES = ("libxrt_coreutil.so.2", "libxrt_core.so.2")
DUMMY_RUNNER = "libvart-dummy-runner.so"
CHECK_TIMEOUT = 120


class Tensor(typing.NamedTuple):
    name: str
    shape: tuple[int, ...]  # without the batch: height, width, channels
    size: int  # values per frame
    fix_point: int  # value = integer * 2**-fix_point
    data_type: str

    @classmethod
    def _from(cls, values) -> "Tensor":
        name, shape, size, fix_point, data_type = values
        return cls(name, tuple(shape), size, fix_point, data_type)


class ModelInfo(typing.NamedTuple):
    inputs: list[Tensor]
    outputs: list[Tensor]
    batch: int
    dpu_subgraphs: int
    devices: list[str]  # device of every subgraph: USER, DPU, CPU
    target: str  # DPU the model was compiled for, for instance DPUCZDX8G_ISA1_B4096
    fingerprint: int


class Core(typing.NamedTuple):
    name: str
    fingerprint: int
    batch: int


def inspect(path: typing.Union[str, os.PathLike]) -> ModelInfo:
    """Structure of a .xmodel (no DPU needed)."""
    info = _opennve_vitis.inspect(os.fspath(path))
    return ModelInfo(
        inputs=[Tensor._from(t) for t in info["inputs"]],
        outputs=[Tensor._from(t) for t in info["outputs"]],
        batch=info["batch"],
        dpu_subgraphs=info["dpu_subgraphs"],
        devices=list(info["devices"]),
        target=info["target"],
        fingerprint=info["fingerprint"],
    )


# ---------------------------------------------------------------------------- the board
def firmware() -> typing.Optional[str]:
    """The DPU design (.xclbin) VART uses: XLNX_VART_FIRMWARE, else "firmware:" of /etc/vart.conf."""
    path = os.environ.get("XLNX_VART_FIRMWARE")
    if path:
        return path
    try:
        for line in pathlib.Path(VART_CONF).read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "firmware" and value.strip():
                return value.strip()
    except OSError:
        pass
    return None


def xrt_problem() -> typing.Optional[str]:
    """Why XRT can not be used, or None."""
    for name in XRT_LIBRARIES:
        try:
            ctypes.CDLL(name)
        except OSError:
            return f"XRT is not installed ({name} missing)"
    return None


def zocl_loaded(root: str = "") -> bool:
    """Is the XRT kernel driver for embedded boards (zocl) loaded and bound to a device?"""
    driver = pathlib.Path(root + "/sys/bus/platform/drivers/zocl")
    return driver.is_dir() and any(entry.is_symlink() and entry.name != "module" for entry in driver.iterdir())


def device_problem() -> typing.Optional[str]:
    """Why the DPU can not be used (checked without VART), or None."""
    if not sys.platform.startswith("linux"):
        return "the DPU runs only on Linux (AMD Kria / Zynq UltraScale+ boards)"
    problem = xrt_problem()
    if problem:
        return problem
    if not zocl_loaded():
        return "the zocl driver is not loaded: no DPU design loaded (sudo xmutil loadapp <design>)"
    path = firmware()
    if not path:
        return f"no DPU design configured ({VART_CONF} without firmware:)"
    if not os.path.isfile(path):
        return f"the DPU design {path} of {VART_CONF} does not exist"
    return None


def _probe(*args: str, timeout: float = CHECK_TIMEOUT) -> dict:
    """Runs opennve_vitis._probe in a separate Python process (VART may stop the process)."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-m", "opennve_vitis._probe", *args],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "GLOG_logtostderr": "1"},
        )
    except subprocess.TimeoutExpired:
        return {"error": f"the check did not finish within {timeout:.0f} s"}
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode == 0 and lines:
        return json.loads(lines[-1])
    if lines:
        data = json.loads(lines[-1])
        if "error" in data:
            return data
    return {"error": _vart_message(result.stderr) or f"the check stopped (exit code {result.returncode})"}


def _vart_message(stderr: str) -> str:
    """The reason in VART's output (glog: "Check failed: ..." / "F... file:line] message")."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in reversed(lines):
        if "Check failed" in line or line.startswith("F"):
            return line.split("] ", 1)[-1]
    for line in reversed(lines):
        if "what():" in line:
            return line.split("what():", 1)[1].strip()
    return lines[-1] if lines else ""


def dpu_cores() -> list[Core]:
    """The DPU cores of the loaded design (checked in a separate process). Raises RuntimeError if
    the DPU can not be used."""
    problem = device_problem()
    if problem:
        raise RuntimeError(problem)
    data = _probe("cores")
    if "error" in data:
        raise RuntimeError(data["error"])
    return [Core(core["name"], core["fingerprint"], core["batch"]) for core in data["cores"]]


def status() -> dict[str, typing.Any]:
    """{"version", "vart", "xrt", "zocl", "firmware", "cores": [{"name", "fingerprint", "batch"}],
    "error"} for opennve system / opennve vitis."""
    info: dict[str, typing.Any] = {
        "version": __version__,
        "vart": VART_VERSION,
        "xrt": xrt_problem() is None,
        "zocl": zocl_loaded() if sys.platform.startswith("linux") else False,
        "firmware": firmware(),
        "cores": [],
        "error": None,
    }
    try:
        info["cores"] = [core._asdict() for core in dpu_cores()]
        if not info["cores"]:
            info["error"] = "the loaded design has no DPU"
    except RuntimeError as error:
        info["error"] = str(error)
    return info


_CHECKED: dict[tuple[str, float, int], typing.Optional[str]] = {}
_CHECK_LOCK = threading.Lock()


def check_model(path: typing.Union[str, os.PathLike]) -> None:
    """Loads the model once in a separate process (cached per file): raises RuntimeError with VART's
    reason if the model does not run on this DPU, instead of VART stopping this process."""
    path = os.path.abspath(os.fspath(path))
    stat = os.stat(path)
    key = (path, stat.st_mtime, stat.st_size)
    with _CHECK_LOCK:
        if key not in _CHECKED:
            problem = device_problem()
            if problem is None:
                data = _probe("check", path)
                problem = data.get("error")
            _CHECKED[key] = problem
        problem = _CHECKED[key]
    if problem:
        raise RuntimeError(f"{os.path.basename(path)}: {problem}")


class Model:
    """A compiled model (.xmodel) on the DPU.

    run() may be called from several threads (calls on one model run one after the other; use one
    model per thread to keep several DPU cores busy). runner: another VART runner library for the
    DPU subgraph, for instance "libvart-dummy-runner.so" (runs nothing: outputs are zero; for
    tests without a DPU). check: test the model in a separate process first (see check_model).
    """

    def __init__(self, path: typing.Union[str, os.PathLike], runner: typing.Optional[str] = None, check: bool = True):
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"model not found: {self.path}")
        self.info = inspect(self.path)
        if self.info.dpu_subgraphs != 1:
            # the same message as the runtime gives, without loading
            _opennve_vitis._Model(self.path, DUMMY_RUNNER)
        if check and not runner:
            check_model(self.path)
        self._model = _opennve_vitis._Model(self.path, runner or "")
        self.input = Tensor._from(self._model.input)
        self.outputs = [Tensor._from(output) for output in self._model.outputs]
        self.batch_size: int = self._model.batch_size

    @property
    def target(self) -> str:
        return self.info.target

    def run(self, batch: np.ndarray) -> list[np.ndarray]:
        """Runs a batch (frames x height x width x channels, or one frame) and returns one
        float32 array per output (frames x height x width x channels)."""
        batch = np.asarray(batch, dtype=np.float32)
        if batch.shape == self.input.shape:
            batch = batch[np.newaxis]
        if batch.shape[1:] != self.input.shape:
            raise ValueError(f"input shape {batch.shape[1:]} != {self.input.shape} of {os.path.basename(self.path)}")
        batch = np.ascontiguousarray(batch)
        count = len(batch)
        buffer = np.empty(sum(output.size for output in self.outputs) * count, np.float32)
        if count:
            self._model.run_into(batch, count, buffer)
        results, offset = [], 0
        for output in self.outputs:
            size = output.size * count
            values = buffer[offset : offset + size]
            offset += size
            results.append(values.reshape((count,) + output.shape))
        return results

    def close(self) -> None:
        self._model.close()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"Model({os.path.basename(self.path)!r}, input={self.input.shape}, "
                f"outputs={[o.shape for o in self.outputs]}, target={self.target!r})")


def write_test_model(path: typing.Union[str, os.PathLike], height: int, width: int, channels: int,
                     fix_in: int = 4, fix_out: int = 3) -> None:
    """Writes a small .xmodel with one DPU subgraph (a ReLU): for tests with runner=DUMMY_RUNNER.
    It has no DPU code and does not run on a real DPU."""
    _opennve_vitis.write_test_model(os.fspath(path), height, width, channels, fix_in, fix_out)
