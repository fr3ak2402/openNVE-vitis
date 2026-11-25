#!/usr/bin/env bash
# Builds the Vitis AI Runtime (unilog, xir, target-factory, vart with the DPU
# runner for edge boards) and the libraries it needs (abseil, protobuf, glog)
# into one prefix. The Python package bundles the result, so the board needs
# nothing of Vitis AI installed, only XRT (libxrt_core / libxrt_coreutil, which
# must match the zocl kernel driver and therefore come from the system).
#
# Usage: ci/build_vart.sh PREFIX [WORKDIR]
#
# Environment:
#   XRT_INCLUDE   directory that contains xrt/ (default /usr/include)
#   XRT_LIBDIR    directory with libxrt_core.so.2 and libxrt_coreutil.so.2
#   JOBS          parallel jobs (default: number of CPUs)
#   VITIS_AI_REF  commit of github.com/Xilinx/Vitis-AI (default below)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

PREFIX=$(realpath -m "${1:?usage: build_vart.sh PREFIX [WORKDIR]}")
WORK=$(realpath -m "${2:-$PREFIX.work}")
JOBS=${JOBS:-$(nproc)}
XRT_INCLUDE=${XRT_INCLUDE:-/usr/include}
XRT_LIBDIR=${XRT_LIBDIR:-/usr/lib/$(gcc -print-multiarch 2>/dev/null || echo)}

# Vitis AI runtime 5.0 (master after the 2025.1 update, Apache 2.0)
VITIS_AI_REF=${VITIS_AI_REF:-77cb9e6ad6749de55cf6de8d4959b1cb4b27020e}
ABSL_TAG=20240722.0
PROTOBUF_TAG=v28.3
GLOG_TAG=v0.6.0 # xir needs google::LogMessageVoidify (removed in 0.7)

# a finished build of the same scripts (for instance from the CI cache) is used as it is
STAMP="$(cat "$HERE/build_vart.sh" "$HERE/patch_vart.py" | sha256sum | cut -c1-16)"
if [[ "$(cat "$PREFIX/.opennve-vitis-build" 2>/dev/null || true)" == "$STAMP" ]]; then
  echo "Vitis AI Runtime in $PREFIX is up to date"
  exit 0
fi
mkdir -p "$PREFIX" "$WORK"
cd "$WORK"

log() { printf '\n== %s\n' "$*"; }

fetch() { # name url ref
  [ -d "$1" ] && return
  git init -q "$1"
  git -C "$1" remote add origin "$2"
  git -C "$1" fetch -q --depth 1 origin "$3"
  git -C "$1" checkout -q FETCH_HEAD
}

log "sources"
fetch absl https://github.com/abseil/abseil-cpp "refs/tags/$ABSL_TAG"
fetch protobuf https://github.com/protocolbuffers/protobuf "refs/tags/$PROTOBUF_TAG"
fetch glog https://github.com/google/glog "refs/tags/$GLOG_TAG"
if [ ! -d vitis-ai ]; then
  git init -q vitis-ai
  git -C vitis-ai remote add origin https://github.com/Xilinx/Vitis-AI
  git -C vitis-ai config core.sparseCheckout true
  echo "src/vai_runtime/" > vitis-ai/.git/info/sparse-checkout
  git -C vitis-ai fetch -q --depth 1 --filter=blob:none origin "$VITIS_AI_REF"
  git -C vitis-ai checkout -q FETCH_HEAD
  # warnings of newer compilers are not errors for us
  find vitis-ai/src/vai_runtime \( -name '*.cmake' -o -name CMakeLists.txt \) \
    -exec sed -i 's/-Werror//g' {} +
fi
python3 "$HERE/patch_vart.py" vitis-ai/src/vai_runtime
RT=$WORK/vitis-ai/src/vai_runtime

export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"

common=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
  -DCMAKE_INSTALL_LIBDIR=lib
  -DCMAKE_PREFIX_PATH="$PREFIX"
  -DCMAKE_CXX_STANDARD=17
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DBUILD_SHARED_LIBS=ON
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DCMAKE_INSTALL_RPATH='$ORIGIN'
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
)

build() { # name source-dir extra-args...
  local name=$1 src=$2
  shift 2
  if [ -f "$WORK/build-$name/.done" ]; then return; fi
  log "$name"
  cmake -S "$src" -B "$WORK/build-$name" "${common[@]}" "$@" > "$WORK/$name.configure.log" 2>&1 ||
    { tail -40 "$WORK/$name.configure.log"; exit 1; }
  cmake --build "$WORK/build-$name" -j "$JOBS" > "$WORK/$name.build.log" 2>&1 ||
    { grep -E 'error|Error' "$WORK/$name.build.log" | head -40; exit 1; }
  cmake --install "$WORK/build-$name" > /dev/null
  touch "$WORK/build-$name/.done"
}

build absl absl -DABSL_PROPAGATE_CXX_STD=ON -DBUILD_TESTING=OFF
build protobuf protobuf -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package \
  -Dprotobuf_BUILD_LIBUPB=OFF -Dprotobuf_INSTALL=ON
build glog glog -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DWITH_UNWIND=OFF -DBUILD_TESTING=OFF

vitis=(-DBUILD_TEST=OFF -DBUILD_PYTHON=OFF -DIS_EDGE=ON
  -DProtobuf_PROTOC_EXECUTABLE="$PREFIX/bin/protoc")
build unilog "$RT/unilog" "${vitis[@]}"
build xir "$RT/xir" "${vitis[@]}"
build target-factory "$RT/target_factory" "${vitis[@]}"
build vart "$RT/vart" "${vitis[@]}" \
  -DENABLE_DPU_RUNNER=ON -DENABLE_CPU_RUNNER=OFF -DENABLE_SIM_RUNNER=OFF \
  -DENABLE_XRNN_RUNNER=OFF -DENABLE_RNN_RUNNER=OFF \
  -DXRT_FOUND=OFF \
  -DCMAKE_MODULE_PATH="$RT/vart/cmake" \
  -DCMAKE_INCLUDE_PATH="$XRT_INCLUDE" -DCMAKE_LIBRARY_PATH="$XRT_LIBDIR"

# licenses of everything the Python package bundles
licenses="$PREFIX/share/licenses/opennve-vitis"
mkdir -p "$licenses"
cp "$RT/vart/Copyright.txt" "$licenses/vitis-ai-runtime.txt"
cp absl/LICENSE "$licenses/abseil.txt"
cp protobuf/LICENSE "$licenses/protobuf.txt"
cp glog/COPYING "$licenses/glog.txt"

echo "$STAMP" > "$PREFIX/.opennve-vitis-build"
log "done: $PREFIX"
ls "$PREFIX/lib" | grep -E '^lib(vart|xir|unilog|target)' | sed 's/^/  /'
