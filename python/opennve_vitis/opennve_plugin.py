"""openNVE plugin "vitis": agents with compiled Vitis AI models (.xmodel) on the DPU of AMD Kria
boards, `opennve vitis`, and the DPU section of `opennve system`.

openNVE finds it through the entry point "opennve.plugins" (pyproject.toml) and imports this
module only then; opennve_vitis itself does not need openNVE.

    opennve plugin install vitis     # installs opennve_vitis (and on a Kria board XRT and the DPU design)
    opennve vitis                    # XRT, zocl driver, DPU design, DPU cores
"""

from __future__ import annotations

import pathlib
import typing

import numpy as np


def plugin():
    """The plugin description for openNVE (called by openNVE when it loads its plugins)."""
    from opennve.plugins import Plugin

    from . import __version__

    return Plugin(
        name="vitis",
        description="DPU of AMD Kria boards (KV260, KR260, K26 / K24): agents with compiled Vitis AI .xmodel models",
        version=__version__,
        register=register,
        commands=commands,
        system_info=system_info,
        system_lines=system_lines,
    )


# ---------------------------------------------------------------------------- backend
def register() -> None:
    from opennve.agents import register_backend

    # one model (DPU runner) per workflow thread, always an accelerator
    register_backend("vitis", create_backend, [".xmodel"], threads="planned", accelerator=True)


def _check_device(device) -> None:
    devices = device if isinstance(device, list) else [device]
    if any(item not in (None, "npu", "auto") for item in devices):
        raise ValueError(
            ".xmodel models run only on the DPU: use \"device\": \"npu\" (or leave it out); "
            "for the CPU / GPU use the .tflite, .keras or PyTorch model the .xmodel was compiled from"
        )


def _backend_class():
    from opennve.agents.backends import Backend, ModelInput

    import opennve_vitis

    class VitisBackend(Backend):
        """Compiled Vitis AI model (.xmodel) on the DPU of an AMD Kria board (DPUCZDX8G).

        The model is quantized (8 bit); the inputs and outputs are converted on the CPU. Inputs and
        outputs are channels last (nhwc). The DPU runs its own batch (usually 1 frame per core);
        several DPU agents and workflow threads share the DPU cores.
        """

        layout = "nhwc"
        device = "npu"
        device_label = "DPU"  # Agent.device_description(): "NPU DPU DPUCZDX8G_ISA1_B4096 (batch 1)"

        def __init__(self, path, device=None, runner: typing.Optional[str] = None):
            _check_device(device)
            self.path = pathlib.Path(path)
            if not self.path.is_file():
                raise FileNotFoundError(f"model not found: {self.path}")
            self._model = opennve_vitis.Model(self.path, runner=runner)
            self.batch_size = self._model.batch_size
            self.input = ModelInput(shape=(None,) + tuple(self._model.input.shape), dtype=np.dtype(np.float32))
            if self._model.target:
                self.device_label = f"DPU {self._model.target}"
            self.device_note = None

        def predict(self, batch: np.ndarray) -> list[np.ndarray]:
            outputs = self._model.run(np.ascontiguousarray(batch, dtype=np.float32))
            # classification heads come as 1 x 1 x classes: give them as (inputs, classes)
            return [output.reshape(len(output), -1) if output.ndim == 4 and output.shape[1:3] == (1, 1) else output
                    for output in outputs]

        def close(self):
            model, self._model = getattr(self, "_model", None), None
            if model is not None:
                try:
                    model.close()
                except Exception:
                    pass

    return VitisBackend


_VITIS_BACKEND = None


def backend_class():
    """The Backend class (created on first use: it derives from openNVE's Backend)."""
    global _VITIS_BACKEND
    if _VITIS_BACKEND is None:
        _VITIS_BACKEND = _backend_class()
    return _VITIS_BACKEND


def create_backend(config):
    """Backend factory for openNVE agents (backend "vitis", .xmodel models)."""
    return backend_class()(config.model_path, config.device)


# ---------------------------------------------------------------------------- system information
def board(root: str = "") -> typing.Optional[str]:
    """Model of the board from the device tree, for instance "ZynqMP SMK-K26 Rev1/B/A", or None."""
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            return pathlib.Path(root + path).read_bytes().rstrip(b"\0").decode(errors="replace").strip() or None
        except OSError:
            continue
    return None


def system_info() -> dict[str, typing.Any]:
    """{"opennve_vitis", "vart", "board", "xrt", "zocl", "firmware", "cores", "error"}"""
    import opennve_vitis

    info = opennve_vitis.status()
    return {
        "opennve_vitis": info["version"],
        "vart": info["vart"],
        "board": board(),
        "xrt": info["xrt"],
        "zocl": info["zocl"],
        "firmware": info["firmware"],
        "cores": info["cores"],
        "error": info["error"],
    }


def system_lines(info: dict) -> list[str]:
    """Lines for `opennve system`."""
    cores = info.get("cores") or []
    if cores:
        names = ", ".join(core["name"] for core in cores)
        state = f"{len(cores)} DPU core{'s' if len(cores) != 1 else ''} ({names})"
    else:
        state = f"DPU not usable: {info.get('error')}"
    return [f"Vitis AI {str(info.get('vart')):<13}{state} (.xmodel models, plugin vitis {info.get('opennve_vitis')})"]


# ---------------------------------------------------------------------------- opennve vitis
def commands(subparsers) -> None:
    command = subparsers.add_parser("vitis", help="show the DPU of the AMD Kria board (plugin vitis)")
    command.add_argument("--json", action="store_true", help="output as JSON")
    command.add_argument("--model", help="also show the structure of a .xmodel (inputs, outputs, DPU)")
    command.set_defaults(function=_status_command)


def _status_command(args) -> int:
    import opennve_vitis

    info = system_info()
    model_info = None
    if getattr(args, "model", None):
        try:
            model_info = opennve_vitis.inspect(args.model)
        except RuntimeError as error:
            model_info = error
    if getattr(args, "json", False):
        import json

        if model_info is not None:
            info["model"] = {"error": str(model_info)} if isinstance(model_info, Exception) else {
                "inputs": [tensor._asdict() for tensor in model_info.inputs],
                "outputs": [tensor._asdict() for tensor in model_info.outputs],
                "batch": model_info.batch, "dpu_subgraphs": model_info.dpu_subgraphs,
                "devices": model_info.devices, "target": model_info.target,
                "fingerprint": model_info.fingerprint,
            }
        print(json.dumps(info, indent=2))
        return 0 if info["cores"] else 1
    print(f"Board                 {info['board'] or 'unknown'}")
    print(f"opennve_vitis         {info['opennve_vitis']} (Vitis AI Runtime {info['vart']})")
    print(f"XRT                   {'installed' if info['xrt'] else 'not installed'}")
    print(f"zocl driver           {'loaded' if info['zocl'] else 'not loaded'}")
    print(f"DPU design            {info['firmware'] or 'not configured (/etc/vart.conf)'}")
    cores = info["cores"]
    if cores:
        for core in cores:
            print(f"DPU core              {core['name']} (fingerprint 0x{core['fingerprint']:x}, batch {core['batch']})")
    else:
        print(f"DPU                   not usable: {info['error']}")
    if model_info is not None:
        print()
        if isinstance(model_info, Exception):
            print(f"Model                 {model_info}")
        else:
            print(f"Model                 {args.model}")
            print(f"Compiled for          {model_info.target or 'unknown'} (fingerprint 0x{model_info.fingerprint:x})")
            print(f"Subgraphs             {' -> '.join(model_info.devices)}")
            for tensor in model_info.inputs:
                print(f"Input                 {tensor.name} {tensor.shape} fix_point {tensor.fix_point}")
            for tensor in model_info.outputs:
                print(f"Output                {tensor.name} {tensor.shape} fix_point {tensor.fix_point}")
            if model_info.dpu_subgraphs != 1:
                print(f"Note                  {model_info.dpu_subgraphs} DPU subgraphs: openNVE-vitis runs models with one")
            fingerprints = {core["fingerprint"] for core in cores}
            if cores and model_info.fingerprint and model_info.fingerprint not in fingerprints:
                print("Note                  the model was compiled for another DPU than the loaded one")
    if not cores:
        print("\nSet up: opennve plugin install vitis (on the Kria board)")
    return 0 if cores else 1
