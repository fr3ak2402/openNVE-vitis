#!/usr/bin/env bash
# Sets up the DPU of an AMD Kria board for openNVE and installs opennve_vitis into the active Python.
#
#     source ~/opennve/.venv/bin/activate    # the Python environment of openNVE
#     bash install.sh                        # or: opennve plugin install vitis
#
# On a Kria board with Ubuntu (22.04, 24.04 or newer) it does everything that is needed, and only
# what is missing:
#   1. finds the board (KV260, KR260, KD240 / K26, K24 SOM)
#   2. installs XRT with the zocl driver from AMD's package sources for Kria (ppa:xilinx-apps,
#      ppa:ubuntu-xilinx/sdk): sudo apt install xrt xrt-dkms; adds the user to the groups render
#      and video (access to the DPU)
#   3. installs the DPU design (FPGA bitstream with the DPU): on the KV260 kv260-benchmark-b4096
#      (DPUCZDX8G B4096, the largest DPU of the KV260) from AMD's kria-apps-firmware, or your own
#      design (--firmware DIR); loads it (xmutil loadapp), loads it at every boot and writes
#      /etc/vart.conf
#   4. installs the ready-made opennve_vitis package (releases of openNVE-vitis, with the Vitis AI
#      Runtime inside) into the active Python
#   5. only if there is none (or with --build): builds the Vitis AI Runtime and opennve_vitis here
#      (needs cmake, g++, Rust; takes about an hour on the board)
# Log out and in again afterwards if the script added you to the groups render / video.
#
# Options:
#     --yes            do not ask before installing system packages / changing the system
#     --no-system      only install opennve_vitis (XRT and the DPU design must be set up)
#     --design NAME    DPU design in /lib/firmware/xilinx/NAME (default on the KV260: kv260-benchmark-b4096)
#     --firmware DIR   install your own DPU design from DIR (.bit.bin or .bin, .dtbo, .xclbin, shell.json)
#                      as /lib/firmware/xilinx/NAME (NAME: --design, default: the name of DIR)
#     --no-boot        do not load the DPU design at every boot
#     --build          build opennve_vitis here instead of installing the ready-made package
#
# Environment variables (optional): PYTHON, CACHE (default ~/.cache/opennve-vitis), see the README.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${OPENNVE_VITIS_ROOT:-}"  # tests: a fake / with proc, sys, etc and lib/firmware
CACHE="${CACHE:-$HOME/.cache/opennve-vitis}"
FIRMWARE_ARCHIVE="${OPENNVE_VITIS_FIRMWARE_ARCHIVE:-https://codeload.github.com/Xilinx/kria-apps-firmware/tar.gz/main}"
RELEASES="${OPENNVE_VITIS_RELEASES:-https://api.github.com/repos/fr3ak2402/openNVE-vitis/releases}"
MIN_XRT="2.13"  # the bundled runtime is built against XRT 2022.1
ASSUME_YES=0
SYSTEM=1
DESIGN=""
FIRMWARE_DIR=""
BOOT=1
BUILD=0
for argument in "$@"; do
    case "$argument" in
        --yes | -y) ASSUME_YES=1 ;;
        --no-system) SYSTEM=0 ;;
        --design=*) DESIGN="${argument#--design=}" ;;
        --firmware=*) FIRMWARE_DIR="${argument#--firmware=}" ;;
        --no-boot) BOOT=0 ;;
        --build) BUILD=1 ;;
        --help | -h) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $argument (--help)" >&2; exit 2 ;;
    esac
done

fail() { echo "error: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }
ask() {
    [[ $ASSUME_YES == 1 ]] && return 0
    local answer
    read -r -p "$1 [Y/n] " answer </dev/tty || answer=""
    [[ -z "$answer" || "$answer" =~ ^[YyJj] ]]
}
installed() { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"; }
available() { [[ -n "$(apt-cache policy "$1" 2>/dev/null | awk '/Candidate:/ {print $2}' | grep -v '(none)')" ]]; }
SUDO=""
if [[ $(id -u) != 0 ]]; then SUDO="sudo"; fi
RELOGIN=0

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="${PYTHON:-$VIRTUAL_ENV/bin/python}"
fi
PYTHON="$(command -v "${PYTHON:-python3}")" || fail "python not found"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' ||
    fail "$PYTHON is older than 3.11 (Ubuntu 22.04 has 3.10: use the Python environment of openNVE)"
[[ "$(uname -s)" == Linux ]] || fail "install.sh is for Linux on AMD Kria boards"

# ---------------------------------------------------------------- 1. board
MODEL="$(tr -d '\0' <"$ROOT/proc/device-tree/model" 2>/dev/null || true)"
CARRIER=""
if [[ $SYSTEM == 1 ]] && command -v xmutil >/dev/null; then
    case "$($SUDO xmutil boardid -b cc 2>/dev/null || true)" in
        *SCK-KV*) CARRIER="kv260" ;;
        *SCK-KR*) CARRIER="kr260" ;;
        *SCK-KD*) CARRIER="kd240" ;;
    esac
fi
if [[ -z "$CARRIER" ]]; then
    case "$MODEL" in
        *KV260* | *kv260* | *SCK-KV*) CARRIER="kv260" ;;
        *KR260* | *kr260* | *SCK-KR*) CARRIER="kr260" ;;
        *KD240* | *kd240* | *SCK-KD*) CARRIER="kd240" ;;
    esac
fi
KRIA=0
[[ -n "$CARRIER" || "$MODEL" == *K26* || "$MODEL" == *K24* || "$MODEL" == *ZynqMP* || "$MODEL" == *Kria* ]] && KRIA=1
step "Board: ${MODEL:-$(uname -m) Linux}${CARRIER:+ (carrier $CARRIER)}"
if [[ $KRIA == 0 && $SYSTEM == 1 ]]; then
    echo "no AMD Kria / Zynq UltraScale+ board found: XRT and the DPU design are not set up (--no-system)"
    SYSTEM=0
fi
if [[ -z "$DESIGN" ]]; then
    if [[ -n "$FIRMWARE_DIR" ]]; then
        DESIGN="$(basename "$FIRMWARE_DIR")"
    elif [[ "$CARRIER" == kv260 || -z "$CARRIER" ]]; then
        DESIGN="kv260-benchmark-b4096"
    fi
fi
FIRMWARE="$ROOT/lib/firmware/xilinx/$DESIGN"

if [[ $SYSTEM == 1 ]]; then
    [[ -n "$DESIGN" ]] || fail "AMD has no ready DPU design for the $CARRIER: build one (Vitis AI DPU TRD) and install it with --firmware=DIR"
    # ------------------------------------------------------------ 2. XRT and zocl
    PACKAGES=()
    installed xrt || PACKAGES+=(xrt)
    installed xrt-dkms || [[ -d "$ROOT/sys/module/zocl" ]] || PACKAGES+=(xrt-dkms)
    if [[ ${#PACKAGES[@]} -gt 0 ]]; then
        step "XRT: sudo apt install ${PACKAGES[*]} (AMD's package sources for Kria)"
        if ask "Install ${PACKAGES[*]} (XRT, zocl driver)?"; then
            if ! grep -rqs "xilinx-apps" "$ROOT/etc/apt/sources.list.d/"; then
                $SUDO add-apt-repository -y ppa:xilinx-apps/ppa
                $SUDO add-apt-repository -y ppa:ubuntu-xilinx/sdk
            fi
            $SUDO apt-get update
            $SUDO apt-get install -y "${PACKAGES[@]}"
        else
            fail "XRT is needed: sudo apt install xrt xrt-dkms"
        fi
    else
        step "XRT: installed"
    fi
    XRT_VERSION="$(dpkg-query -W -f='${Version}' xrt 2>/dev/null | grep -oE '2\.[0-9]+\.[0-9]+' | head -1 || true)"
    if [[ -n "$XRT_VERSION" ]] && ! "$PYTHON" -c "import sys; v = tuple(map(int, '$XRT_VERSION'.split('.')[:2])); sys.exit(0 if v >= tuple(map(int, '$MIN_XRT'.split('.'))) else 1)"; then
        fail "XRT $XRT_VERSION is too old (needs $MIN_XRT or newer, 2022.1): add ppa:xilinx-apps/ppa and ppa:ubuntu-xilinx/sdk, then sudo apt install xrt xrt-dkms"
    fi
    USER_NAME="${SUDO_USER:-$(id -un)}"
    for group in render video; do
        if getent group "$group" >/dev/null && ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$group"; then
            step "Access to the DPU: adding $USER_NAME to the group $group"
            $SUDO usermod -aG "$group" "$USER_NAME"
            RELOGIN=1
        fi
    done

    # ------------------------------------------------------------ 3. DPU design
    if [[ -n "$FIRMWARE_DIR" ]]; then
        [[ -d "$FIRMWARE_DIR" ]] || fail "--firmware: $FIRMWARE_DIR is not a directory"
        ls "$FIRMWARE_DIR"/*.xclbin >/dev/null 2>&1 || fail "--firmware: $FIRMWARE_DIR has no .xclbin"
        step "DPU design: $FIRMWARE_DIR -> $FIRMWARE"
        $SUDO mkdir -p "$FIRMWARE"
        $SUDO cp "$FIRMWARE_DIR"/* "$FIRMWARE/"
    elif ! ls "$FIRMWARE"/*.xclbin >/dev/null 2>&1; then
        step "DPU design: $DESIGN (AMD kria-apps-firmware)"
        ask "Install the DPU design $DESIGN?" || fail "a DPU design is needed (--firmware=DIR for your own)"
        if available "xlnx-firmware-$DESIGN"; then
            $SUDO apt-get install -y "xlnx-firmware-$DESIGN"
        else
            # build the firmware files (bitstream -> .bin, device tree overlay) from AMD's repository
            NEED=()
            command -v bootgen >/dev/null || NEED+=(bootgen-xlnx)
            command -v dtc >/dev/null || NEED+=(device-tree-compiler)
            command -v make >/dev/null || NEED+=(make)
            [[ ${#NEED[@]} -eq 0 ]] || $SUDO apt-get install -y "${NEED[@]}"
            SOURCE="$CACHE/kria-apps-firmware"
            rm -rf "$SOURCE" && mkdir -p "$SOURCE"
            curl -fsSL "$FIRMWARE_ARCHIVE" | tar -xz -C "$SOURCE" --strip-components=1
            APP="${DESIGN#*-}"  # kv260-benchmark-b4096 -> benchmark-b4096
            BOARD="${DESIGN%%-*}"
            [[ -d "$SOURCE/boards/$BOARD/$APP" ]] || fail "kria-apps-firmware has no design $DESIGN"
            $SUDO make -C "$SOURCE/boards" APPS="$BOARD/$APP/" DESTDIR="$ROOT" install
        fi
    else
        step "DPU design: $DESIGN is installed"
    fi
    XCLBIN="$(ls "$FIRMWARE"/*.xclbin 2>/dev/null | head -1)"
    [[ -n "$XCLBIN" ]] || fail "the DPU design $DESIGN has no .xclbin"
    XCLBIN="${XCLBIN#"$ROOT"}"

    if command -v xmutil >/dev/null; then
        if ! $SUDO xmutil listapps 2>/dev/null | grep -E "^\s*$DESIGN\s" | grep -qE '\s0,?\s*$'; then
            step "DPU design: loading $DESIGN (xmutil loadapp)"
            $SUDO xmutil unloadapp >/dev/null 2>&1 || true
            $SUDO xmutil loadapp "$DESIGN"
        fi
    else
        echo "xmutil not found: load the design yourself (sudo xmutil loadapp $DESIGN after sudo xlnx-config.sysinit)"
    fi
    if [[ $BOOT == 1 && -d "$ROOT/etc/dfx-mgrd" ]]; then
        if [[ "$(cat "$ROOT/etc/dfx-mgrd/default_firmware" 2>/dev/null || true)" != "$DESIGN" ]]; then
            step "DPU design: loaded at every boot (/etc/dfx-mgrd/default_firmware)"
            echo "$DESIGN" | $SUDO tee "$ROOT/etc/dfx-mgrd/default_firmware" >/dev/null
        fi
    fi
    if ! grep -qx "firmware: $XCLBIN" "$ROOT/etc/vart.conf" 2>/dev/null; then
        step "Vitis AI Runtime: /etc/vart.conf -> $XCLBIN"
        echo "firmware: $XCLBIN" | $SUDO tee "$ROOT/etc/vart.conf" >/dev/null
    fi
fi

# ---------------------------------------------------------------- 4. opennve_vitis: ready-made package
# The releases of openNVE-vitis have ready-made packages (Linux aarch64, glibc 2.35: Ubuntu 22.04 and
# newer; every Python from 3.11) with the Vitis AI Runtime inside: no Rust, no Vitis AI needed.
"$PYTHON" -m pip --version >/dev/null 2>&1 || "$PYTHON" -m ensurepip --upgrade >/dev/null
install_wheel() {
    "$PYTHON" -m pip install --quiet "numpy>=1.26"
    "$PYTHON" -m pip install --force-reinstall --no-deps "$1"
}
WHEEL=""
if [[ $BUILD == 0 ]]; then
    step "opennve_vitis: ready-made package ($(uname -m))"
    mkdir -p "$CACHE/wheels"
    WHEEL="$("$PYTHON" - "$(uname -m)" "$CACHE/wheels" "$RELEASES" <<'EOF' || true
import json, pathlib, sys, urllib.request

machine, folder, url = sys.argv[1:4]
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"}), timeout=30) as response:
        releases = json.load(response)
except Exception as error:
    print(f"no connection to the releases ({error})", file=sys.stderr)
    sys.exit(1)
for release in releases:  # newest first
    if release.get("draft"):
        continue
    for asset in release.get("assets", []):
        name = asset["name"]
        if name.startswith("opennve_vitis-") and name.endswith(f"-cp311-abi3-manylinux_2_35_{machine}.whl"):
            target = pathlib.Path(folder) / name
            target.unlink(missing_ok=True)
            urllib.request.urlretrieve(asset["browser_download_url"], target)
            print(f"{release['tag_name']}: {name}", file=sys.stderr)
            print(target)
            sys.exit(0)
print(f"no ready-made package for {machine}", file=sys.stderr)
sys.exit(1)
EOF
)"
fi
if [[ -n "$WHEEL" ]]; then
    install_wheel "$WHEEL"
else
    [[ $BUILD == 1 ]] || echo "building it instead (Vitis AI Runtime and Rust module: about an hour on the board)"

    # ------------------------------------------------------------ 5. build tools, Rust, build
    if [[ $SYSTEM == 1 || $BUILD == 1 ]] && command -v apt-get >/dev/null; then
        MISSING=()
        for package in build-essential cmake ninja-build git curl pkg-config patchelf libboost-dev \
            libboost-filesystem-dev uuid-dev zlib1g-dev; do
            installed "$package" || MISSING+=("$package")
        done
        installed libxrt-dev || installed xrt || MISSING+=(libxrt-dev)
        if [[ ${#MISSING[@]} -gt 0 ]]; then
            step "Build tools: sudo apt install ${MISSING[*]}"
            ask "Install ${MISSING[*]}?" && $SUDO apt-get install -y "${MISSING[@]}"
        fi
    fi
    if ! command -v cargo >/dev/null && [[ -x "$HOME/.cargo/bin/cargo" ]]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    if ! command -v cargo >/dev/null; then
        step "Rust: rustup (for $(id -un), in ~/.cargo)"
        if ask "Install Rust with rustup?"; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            fail "Rust is needed to build opennve_vitis"
        fi
    fi
    step "Vitis AI Runtime: build into $CACHE/vart"
    bash "$HERE/ci/build_vart.sh" "$CACHE/vart" "$CACHE/vart-work"

    step "opennve_vitis: build and install into $PYTHON"
    "$PYTHON" -m pip install --quiet --upgrade "maturin>=1.8,<2"
    rm -rf "$CACHE/dist" "$CACHE/repaired"
    (cd "$HERE/python" && VITIS_AI_PREFIX="$CACHE/vart" OPENNVE_VITIS_NO_RPATH=1 \
        "$PYTHON" -m maturin build --release --interpreter "$PYTHON" --out "$CACHE/dist")
    "$PYTHON" "$HERE/ci/repair_wheel.py" "$(ls "$CACHE"/dist/opennve_vitis-*.whl | head -1)" "$CACHE/vart" "$CACHE/repaired"
    install_wheel "$(ls "$CACHE"/repaired/opennve_vitis-*.whl | head -1)"
fi

"$PYTHON" - <<'EOF'
import opennve_vitis

print(f"opennve_vitis {opennve_vitis.__version__} with the Vitis AI Runtime {opennve_vitis.VART_VERSION}")
status = opennve_vitis.status()
if status["cores"]:
    print("DPU:", ", ".join(f"{core['name']} (fingerprint 0x{core['fingerprint']:x})" for core in status["cores"]))
else:
    print("DPU: not usable yet:", status["error"])
EOF

if [[ $RELOGIN == 1 ]]; then
    echo
    echo "Log out and in again (or reboot): the groups render / video give access to the DPU."
fi
