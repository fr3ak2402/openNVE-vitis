"""Tests without a DPU: a generated .xmodel, run with VART's dummy runner (outputs are zero), and
the checks of the board."""

import os
import subprocess
import sys

import numpy as np
import pytest

import opennve_vitis


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "test.xmodel"
    opennve_vitis.write_test_model(path, 8, 6, 2, fix_in=4, fix_out=3)
    return path


def test_versions():
    assert opennve_vitis.__version__
    assert opennve_vitis.VART_VERSION.count(".") == 2


def test_inspect(model_path):
    info = opennve_vitis.inspect(model_path)
    assert info.dpu_subgraphs == 1
    assert info.devices == ["USER", "DPU"]
    assert info.target == "DPUCZDX8G_ISA1_B4096"
    assert info.inputs == [opennve_vitis.Tensor("input", (8, 6, 2), 96, 4, "XINT8")]
    assert info.outputs[0].shape == (8, 6, 2) and info.outputs[0].fix_point == 3
    assert info.batch == 1


def test_inspect_errors(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        opennve_vitis.inspect(tmp_path / "missing.xmodel")


def test_dummy_runner(model_path):
    with opennve_vitis.Model(model_path, runner=opennve_vitis.DUMMY_RUNNER) as model:
        assert model.input.shape == (8, 6, 2)
        assert model.target == "DPUCZDX8G_ISA1_B4096"
        outputs = model.run(np.random.default_rng(0).random((5, 8, 6, 2), np.float32))
        assert len(outputs) == 1
        assert outputs[0].shape == (5, 8, 6, 2) and outputs[0].dtype == np.float32
        assert model.run(np.zeros((8, 6, 2), np.float32))[0].shape == (1, 8, 6, 2)
        assert model.run(np.zeros((0, 8, 6, 2), np.float32))[0].shape == (0, 8, 6, 2)
        with pytest.raises(ValueError, match="input shape"):
            model.run(np.zeros((2, 8, 6, 3), np.float32))
        assert "test.xmodel" in repr(model)


def test_model_needs_a_dpu_without_runner(model_path, monkeypatch):
    """Without DPU the model is refused before VART is used (no crash of the process)."""
    monkeypatch.setattr(opennve_vitis, "device_problem", lambda: "no DPU here")
    opennve_vitis._CHECKED.clear()
    with pytest.raises(RuntimeError, match="no DPU here"):
        opennve_vitis.Model(model_path)


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        opennve_vitis.Model(tmp_path / "missing.xmodel")


def test_firmware(tmp_path, monkeypatch):
    conf = tmp_path / "vart.conf"
    conf.write_text("firmware: /lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin\n")
    monkeypatch.setattr(opennve_vitis, "VART_CONF", str(conf))
    monkeypatch.delenv("XLNX_VART_FIRMWARE", raising=False)
    assert opennve_vitis.firmware().endswith("kv260-benchmark-b4096.xclbin")
    monkeypatch.setenv("XLNX_VART_FIRMWARE", "/tmp/other.xclbin")
    assert opennve_vitis.firmware() == "/tmp/other.xclbin"


def test_zocl(tmp_path):
    driver = tmp_path / "sys/bus/platform/drivers/zocl"
    driver.mkdir(parents=True)
    (driver / "module").symlink_to(tmp_path)
    assert not opennve_vitis.zocl_loaded(str(tmp_path))
    (driver / "axi:zyxclmm_drm").symlink_to(tmp_path)
    assert opennve_vitis.zocl_loaded(str(tmp_path))


def test_status_without_dpu(monkeypatch):
    monkeypatch.setattr(opennve_vitis, "device_problem", lambda: "no DPU here")
    status = opennve_vitis.status()
    assert status["vart"] == opennve_vitis.VART_VERSION
    assert status["cores"] == [] and status["error"] == "no DPU here"


def test_probe_failure_becomes_an_error():
    """A failing check process (here: a wrong command) gives an error message, not an exception."""
    data = opennve_vitis._probe("nonsense")
    assert data["error"]


def test_probe_check_reports_errors(tmp_path):
    broken = tmp_path / "broken.xmodel"
    broken.write_bytes(b"not a model")
    data = opennve_vitis._probe("check", str(broken))
    assert data["error"]


def test_vart_message():
    stderr = "I1124 10:00:00 log\nF1124 10:00:01.123 123 dpu_session.cpp:42] Check failed: fingerprint mismatch\n"
    assert opennve_vitis._vart_message(stderr) == "Check failed: fingerprint mismatch"


def test_bundled_runtime_is_used():
    """The module loads the runtime from opennve_vitis/lib, not from the system."""
    lib = os.path.join(os.path.dirname(opennve_vitis.__file__), "lib")
    if not os.path.isdir(lib):
        pytest.skip("development build (runtime not bundled)")
    output = subprocess.run(["ldd", opennve_vitis._opennve_vitis.__file__], capture_output=True, text=True).stdout
    vart_lines = [line for line in output.splitlines() if "libvart-runner" in line or "libxir" in line]
    assert vart_lines and all(lib in line for line in vart_lines)
    assert os.path.isfile(os.path.join(lib, "libvart-dpu-runner.so"))
    assert os.path.isdir(os.path.join(lib, "licenses"))


def test_import_in_clean_process():
    code = "import opennve_vitis; print(opennve_vitis.VART_VERSION)"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                            env={k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"})
    assert result.returncode == 0, result.stderr
