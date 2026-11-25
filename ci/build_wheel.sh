#!/usr/bin/env bash
# Builds the ready-made opennve_vitis package inside an Ubuntu 22.04 container (glibc 2.35: runs
# on Kria Ubuntu 22.04, 24.04 and newer):
#
#     docker run --rm -v "$PWD:/io" ubuntu:22.04 bash /io/ci/build_wheel.sh
#
# The package contains the Vitis AI Runtime (built from source by ci/build_vart.sh, against the
# XRT 2.13 headers so that it runs with every newer XRT of the board) but not XRT itself.
# Output: /io/dist/opennve_vitis-<version>-cp311-abi3-manylinux_2_35_<machine>.whl
set -euo pipefail
IO="${IO:-/io}"
MACHINE="$(uname -m)"
ARCH="$(dpkg --print-architecture)" # arm64 / amd64
CACHE="$IO/.cache/$MACHINE"
# XRT 2022.1 (2.13) from Ubuntu 24.04 (universe), only to compile and link against
XRT_DEB_VERSION="202210.2.13.466+dfsg-8ubuntu5"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates curl python3 pkg-config file patchelf \
    libboost-dev libboost-filesystem-dev libboost-system-dev uuid-dev zlib1g-dev >/dev/null

TRIPLE="$(gcc -print-multiarch)"
XRT=/opt/xrt
if [[ ! -f "$XRT/usr/include/xrt/version.h" ]]; then
    case "$ARCH" in
        arm64) MIRROR=http://ports.ubuntu.com/ubuntu-ports ;;
        *) MIRROR=http://archive.ubuntu.com/ubuntu ;;
    esac
    mkdir -p "$XRT" /tmp/xrt-debs
    for package in libxrt-dev libxrt1; do
        deb="/tmp/xrt-debs/${package}_${XRT_DEB_VERSION}_${ARCH}.deb"
        curl -fsSL -o "$deb" "$MIRROR/pool/universe/x/xrt/${package}_${XRT_DEB_VERSION/+/%2B}_${ARCH}.deb"
        dpkg -x "$deb" "$XRT"
    done
fi

# Vitis AI Runtime (cached between CI runs: .cache/<machine>/vart)
JOBS="$(nproc)" XRT_INCLUDE="$XRT/usr/include" XRT_LIBDIR="$XRT/usr/lib/$TRIPLE" \
    bash "$IO/ci/build_vart.sh" "$CACHE/vart" "$CACHE/vart-work"

# Rust and maturin (uv brings a Python >= 3.11; Ubuntu 22.04 has 3.10)
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
    source "$HOME/.cargo/env"
fi
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
fi
uv python install -q 3.12
PYTHON="$(uv python find 3.12)"

export VITIS_AI_PREFIX="$CACHE/vart" OPENNVE_VITIS_NO_RPATH=1
export CARGO_TARGET_DIR="$CACHE/target"
# Rust tests here, where the runtime's libraries (boost of Ubuntu 22.04) are
(cd "$IO" && LD_LIBRARY_PATH="$CACHE/vart/lib" cargo test -p vitis-runtime)
OUT=/tmp/wheels
rm -rf "$OUT"
(cd "$IO/python" && uvx --python "$PYTHON" "maturin>=1.8,<2" build --release --interpreter "$PYTHON" \
    --compatibility linux --auditwheel skip --out "$OUT")
mkdir -p "$IO/dist"
python3 "$IO/ci/repair_wheel.py" "$OUT"/opennve_vitis-*.whl "$CACHE/vart" "$IO/dist" \
    --plat "manylinux_2_35_$MACHINE"
ls -l "$IO/dist"
