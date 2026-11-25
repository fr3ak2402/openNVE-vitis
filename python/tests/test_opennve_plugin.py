"""Tests of the openNVE plugin (backend "vitis", .xmodel agents): with a fake model (to compare
predictions) and with a generated .xmodel on VART's dummy runner (the real path without a DPU).

Skipped if openNVE (with plugins) is not installed.
"""

import importlib.metadata

import numpy as np
import pytest

opennve = pytest.importorskip("opennve")
pytest.importorskip("opennve.plugins")

import opennve_vitis  # noqa: E402
from opennve_vitis import opennve_plugin  # noqa: E402

WEIGHTS = np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], np.float32)  # 1 x 1 convolution, 2 -> 3 classes
BIAS = np.array([1.0, 0.0, 0.0], np.float32)
PREDICTION = {"task": "segmentation", "activation": "softmax", "target_classes": [1, 2]}


class FakeModel:
    """Stands in for opennve_vitis.Model: the .xmodel "model" is a 1 x 1 convolution."""

    instances: list = []

    def __init__(self, path, runner=None, check=True):
        if str(path).endswith("other_dpu.xmodel"):
            raise RuntimeError("other_dpu.xmodel: compiled for DPUCZDX8G_ISA1_B1024, the loaded design has DPUCZDX8G:1")
        self.batch_size = 1
        self.input = opennve_vitis.Tensor("input", (16, 16, 2), 512, 4, "XINT8")
        self.outputs = [opennve_vitis.Tensor("output", (16, 16, 3), 768, 2, "XINT8")]
        self.target = "DPUCZDX8G_ISA1_B4096"
        self.closed = False
        FakeModel.instances.append(self)

    def run(self, batch):
        assert batch.dtype == np.float32 and batch.flags["C_CONTIGUOUS"] and batch.shape[1:] == (16, 16, 2)
        return [batch @ WEIGHTS + BIAS]

    def close(self):
        self.closed = True


@pytest.fixture
def plugin(monkeypatch):
    from opennve import plugins

    entry_point = importlib.metadata.EntryPoint("vitis", "opennve_vitis.opennve_plugin:plugin", plugins.ENTRY_POINT_GROUP)
    monkeypatch.setattr(plugins, "_entry_points", lambda: [entry_point])
    plugins._reset()
    FakeModel.instances.clear()
    yield
    plugins._reset()


@pytest.fixture
def fake(plugin, monkeypatch):
    monkeypatch.setattr(opennve_vitis, "Model", FakeModel)


def _frame(width=40, height=24):
    rng = np.random.default_rng(0)
    events = np.zeros(500, dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])
    events["t"] = np.sort(rng.integers(0, 40_000, 500))
    events["x"] = rng.integers(0, width, 500)
    events["y"] = rng.integers(0, height, 500)
    events["on"] = rng.random(500) < 0.6
    window = opennve.EventWindow(camera="left", start_t=0, end_t=40_000, width=width, height=height,
                                 polarity_events=events)
    return window.frame({"mode": "count", "tiles": {"size": 16, "min_events": 0}})


def test_plugin_is_found(plugin):
    opennve.plugins.load_plugins()
    installed = opennve.plugins.installed_plugins()["vitis"]
    assert installed.error is None and installed.version == opennve_vitis.__version__
    assert opennve.agents.backend_options("vitis") == {"threads": "planned", "accelerator": True}


def test_xmodel_agent(fake, tmp_path):
    (tmp_path / "unet.xmodel").write_bytes(b"fake")
    config = opennve.AgentConfig(model="unet.xmodel", base_dir=tmp_path, prediction=PREDICTION)
    assert config.resolved_backend == "vitis"
    agent = opennve.Agent(config)
    assert agent.layout == "nhwc" and agent.device == "npu"
    assert agent.device_description().startswith("NPU DPU")
    frame = _frame()
    result = agent.predict(frame)
    reference = opennve.Agent.from_model(lambda batch: batch @ WEIGHTS + BIAS, prediction=PREDICTION).predict(frame)
    for a, b in zip(result.predictions, reference.predictions):
        np.testing.assert_array_equal(a.class_map, b.class_map)
    agent.close()
    assert FakeModel.instances[0].closed


def test_devices_and_errors(fake, tmp_path):
    (tmp_path / "a.xmodel").write_bytes(b"fake")
    for device in ("cpu", "gpu", ["npu", "cpu"]):
        with pytest.raises(ValueError, match="only on the DPU"):
            opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "a.xmodel"), device=device)).load()
    (tmp_path / "other_dpu.xmodel").write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="compiled for"):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "other_dpu.xmodel"))).load()
    with pytest.raises(FileNotFoundError):
        opennve.Agent(opennve.AgentConfig(model=str(tmp_path / "missing.xmodel"))).load()


def test_classification_head(fake, tmp_path, monkeypatch):
    class Head(FakeModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.input = opennve_vitis.Tensor("input", (1, 1, 2), 2, 4, "XINT8")

        def run(self, batch):
            return [np.ones((len(batch), 1, 1, 3), np.float32)]

    monkeypatch.setattr(opennve_vitis, "Model", Head)
    (tmp_path / "c.xmodel").write_bytes(b"fake")
    backend = opennve_plugin.backend_class()(tmp_path / "c.xmodel")
    assert backend.predict(np.ones((5, 1, 1, 2), np.float32))[0].shape == (5, 3)


def test_generated_xmodel_on_dummy_runner(plugin, tmp_path, monkeypatch):
    """The real path (VART runner, quantization) with the dummy runner: outputs are zero."""
    opennve_vitis.write_test_model(tmp_path / "relu.xmodel", 16, 16, 2)
    real = opennve_vitis.Model
    monkeypatch.setattr(opennve_vitis, "Model", lambda path, runner=None, check=True: real(path, runner=opennve_vitis.DUMMY_RUNNER))
    agent = opennve.Agent(opennve.AgentConfig(model="relu.xmodel", base_dir=tmp_path,
                                              prediction={"task": "segmentation", "activation": "softmax"}))
    assert agent.device_description() == "NPU DPU DPUCZDX8G_ISA1_B4096 (batch 1)"
    result = agent.predict(_frame())
    assert result.predictions
    agent.close()


def test_cli_and_system(plugin, capsys, monkeypatch, tmp_path):
    from opennve.cli import main

    assert main(["plugin", "list"]) == 0
    assert "vitis" in capsys.readouterr().out
    monkeypatch.setattr(opennve_vitis, "device_problem", lambda: "no DPU here")
    opennve_vitis.write_test_model(tmp_path / "relu.xmodel", 16, 16, 2)
    assert main(["vitis", "--model", str(tmp_path / "relu.xmodel")]) == 1
    out = capsys.readouterr().out
    assert "opennve_vitis" in out and "no DPU here" in out and "DPUCZDX8G_ISA1_B4096" in out
    assert main(["vitis", "--json"]) == 1
    info = opennve.system_info()["plugins"]["vitis"]
    assert info["opennve_vitis"] == opennve_vitis.__version__
    assert opennve_plugin.system_lines(info)[0].startswith("Vitis AI")


def test_board(tmp_path):
    (tmp_path / "proc/device-tree").mkdir(parents=True)
    (tmp_path / "proc/device-tree/model").write_bytes(b"ZynqMP SMK-K26 Rev1/B/A\0")
    assert opennve_plugin.board(str(tmp_path)) == "ZynqMP SMK-K26 Rev1/B/A"
    assert opennve_plugin.board(str(tmp_path / "none")) is None
