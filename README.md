# openNVE-vitis

The DPU of AMD Kria boards for [openNVE](https://github.com/fr3ak2402/openNVE): runs compiled
Vitis AI models (`.xmodel`) with the Vitis AI Runtime (VART). The runtime is called from Rust and
used from Python. It works the same way as the event camera drivers and
[openNVE-hailo](https://github.com/fr3ak2402/openNVE-hailo).

- **`runtime/`**: the Rust crate `vitis-runtime`. It is a bridge (cxx) to the C++ API of VART and
  XIR. It reads models, creates the DPU runner, converts float32 to and from the DPU's 8-bit values
  and runs batches.
- **`python/`**: the Python package `opennve_vitis` (PyO3, numpy). It also contains the openNVE
  plugin `vitis`.
- **`install.sh`**: sets up the board and installs the package (`opennve plugin install vitis`).
- **`ci/`**: builds the Vitis AI Runtime from source and the ready-made package.

**Boards:** the KV260 Vision AI Starter Kit (DPUCZDX8G B4096, AMD's design `kv260-benchmark-b4096`).
The KR260, KD240 and other K26 / K24 or Zynq UltraScale+ boards work with your own DPU design
(`--firmware`).

**System:** Ubuntu 22.04, 24.04 or newer for Kria, on aarch64, with Python 3.11 or newer.

> Status: this was built and tested without a board. Generated models run on VART's dummy runner,
> the Rust and Python tests pass, and `install.sh` was tested on a fake KV260. It has not yet run
> on a real DPU.

## Why the package brings its own Vitis AI Runtime

AMD's Vitis AI packages for the Kria boards are tied to one Ubuntu version and to its Python (3.10
on Ubuntu 22.04). The ready-made `opennve_vitis` package therefore contains the Vitis AI Runtime
5.0:

- unilog, xir, target-factory and vart with the DPU runner
- abseil, protobuf, glog and boost, which the runtime needs

`ci/build_vart.sh` builds all of it from AMD's open source code (Apache 2.0). The package runs with
every Python from 3.11 (abi3) and on every Ubuntu from 22.04 (glibc 2.35).

From the system it needs only **XRT**, because XRT must match the board's kernel driver:

- `libxrt_core` and `libxrt_coreutil`
- the zocl driver

The runtime is compiled against XRT 2.13 (2022.1), the oldest XRT of the Kria package sources, so it
runs with every newer XRT too. `install.sh` installs XRT from AMD's package sources for Kria.

## Installation (on the Kria board)

```bash
source ~/opennve/.venv/bin/activate      # the Python environment of openNVE
opennve plugin install vitis             # or: bash install.sh
opennve vitis                            # XRT, zocl, DPU design, DPU cores
```

`install.sh` does only what is missing, and asks before changing the system (`--yes` skips the
questions):

1. Finds the board: KV260, KR260 or KD240 (`xmutil boardid`, device tree).
2. Installs XRT and the zocl driver: `ppa:xilinx-apps/ppa` and `ppa:ubuntu-xilinx/sdk`, then
   `sudo apt install xrt xrt-dkms`. It checks that XRT is 2.13 or newer. It also adds the user to
   the groups `render` and `video`, which give access to the DPU; log in again afterwards.
3. Installs the DPU design. On the KV260 that is `kv260-benchmark-b4096` from AMD's
   [kria-apps-firmware](https://github.com/Xilinx/kria-apps-firmware): DPUCZDX8G B4096 at
   300 MHz, the largest DPU that fits. It is installed through AMD's apt package if there is one,
   otherwise built with `bootgen` and `dtc`. The script then:
   - loads the design with `xmutil loadapp`
   - has it loaded at every boot (`/etc/dfx-mgrd/default_firmware`)
   - writes `/etc/vart.conf` (`firmware: …xclbin`)
4. Installs the ready-made `opennve_vitis` package from the
   [releases](https://github.com/fr3ak2402/openNVE-vitis/releases) into the active Python.
5. Builds it on the board instead, but only if there is no ready-made package or you pass
   `--build`. This takes about an hour on the board's Cortex-A53.

Options:

| Option | Effect |
| --- | --- |
| `--yes` | Do not ask before changing the system |
| `--no-system` | Install only the package (XRT and the DPU design are already set up) |
| `--design NAME` | Use the design in `/lib/firmware/xilinx/NAME` |
| `--firmware DIR` | Install your own design. DIR holds `.bit.bin` / `.bin`, `.dtbo`, `.xclbin` and `shell.json` |
| `--no-boot` | Do not load the design at boot |
| `--build` | Build here instead of using the ready-made package |

Options for `opennve plugin install` come after `--`, for example
`opennve plugin install vitis -- --yes`.

## Compiling models (on a PC, x86_64)

The DPU runs quantized models (8 bit) that the Vitis AI compiler built for exactly this DPU
configuration. Compile them with the Vitis AI 3.5 Docker image, which has a frozen compiler for the
DPUCZDX8G.

```bash
docker run -it -v $PWD:/workspace xilinx/vitis-ai-tensorflow2-cpu:latest   # PyTorch: vitis-ai-pytorch-cpu
```

```python
# quantize with a few hundred typical frames (calibration), TensorFlow 2 / Keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantized = vitis_quantize.VitisQuantizer(model).quantize_model(calib_dataset=frames)
quantized.save("unet_quantized.h5")
```

```bash
# compile for the DPU of the KV260 (DPUCZDX8G_ISA1_B4096)
vai_c_tensorflow2 -m unet_quantized.h5 -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o out -n unet
# -> out/unet.xmodel
```

Notes:

- **One DPU subgraph.** Layers the DPU does not support run on the CPU and split the model.
  openNVE-vitis runs models with exactly one DPU subgraph. `opennve vitis --model unet.xmodel`
  shows the subgraphs.
- **No final softmax or sigmoid.** openNVE applies the activation from `"prediction"`.
- **One input.** Tensors are channels last: frames × height × width × channels.
- **The model must match the loaded DPU.** The fingerprint of the model must match the DPU.
  Otherwise the model is refused with a message instead of stopping the program.

## openNVE agents

The suffix `.xmodel` selects the backend `vitis`. The model runs on the DPU (`"device": "npu"` or
leave it out):

```json
{
  "name": "unet-dpu",
  "model": "models/unet.xmodel",
  "prediction": {"task": "segmentation", "activation": "softmax", "target_classes": [1, 2]}
}
```

The same agent with `unet.tflite`, `unet.keras` or `unet.pt2` runs on the CPU / GPU. `opennve
system` shows the DPU, and `opennve benchmark` compares the agents.

## Python

```python
import numpy as np, opennve_vitis

opennve_vitis.status()                    # {"xrt": True, "zocl": True, "firmware": ..., "cores": [...]}
opennve_vitis.inspect("unet.xmodel")      # inputs, outputs, subgraphs, target DPU (no DPU needed)
with opennve_vitis.Model("unet.xmodel") as model:
    outputs = model.run(np.zeros((16, 80, 80, 2), np.float32))   # [array (16, 80, 80, 3)]
```

Before a model is loaded, `Model` checks it once in a separate process. This is because VART stops
the whole process on some errors: no DPU loaded, a model for another DPU, a broken file. The check
turns those into a `RuntimeError` with VART's reason.

`Model(path, runner=opennve_vitis.DUMMY_RUNNER)` runs without a DPU, with zero outputs.
`opennve_vitis.write_test_model()` writes a small model for that. The tests use both.

## Rust

```rust
let model = vitis_runtime::Model::load("unet.xmodel")?;
let outputs = model.run(&input, frames)?;   // one Vec<f32> per output
```

## Building from source

```bash
ci/build_vart.sh ~/vart                    # Vitis AI Runtime + abseil, protobuf, glog (needs cmake, ninja, g++, boost, uuid, XRT headers)
export VITIS_AI_PREFIX=~/vart LD_LIBRARY_PATH=~/vart/lib
cargo test -p vitis-runtime
cd python && maturin build --release --out dist
python ../ci/repair_wheel.py dist/opennve_vitis-*.whl ~/vart wheels   # puts the runtime into the package
```

The ready-made package is built by `ci/build_wheel.sh` in an Ubuntu 22.04 container (CI). A tag
`v*` publishes it in a release, where `install.sh` finds it.

## Licenses of the bundled libraries

The package contains, in `opennve_vitis/lib/licenses`:

- Vitis AI Runtime (AMD, Apache 2.0)
- abseil (Apache 2.0)
- protobuf (BSD-3-Clause)
- glog (BSD-3-Clause)
- boost.filesystem (Boost Software License)

XRT is not part of the package.
