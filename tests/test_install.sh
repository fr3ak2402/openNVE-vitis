#!/usr/bin/env bash
# Tests install.sh on a fake Kria KV260 (fake /proc, /sys, /etc, /lib/firmware, and fake sudo /
# apt-get / dpkg-query / xmutil / bootgen / dtc / usermod). The package is the wheel in dist/
# (WHEEL=...), installed into a temporary Python environment.
#
#     WHEEL=dist/opennve_vitis-...whl bash tests/test_install.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL="$(realpath "${WHEEL:-$(ls "$HERE"/dist/opennve_vitis-*.whl | head -1)}")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PYTHON_FOR_TESTS="${PYTHON_FOR_TESTS:-python3}"
"$PYTHON_FOR_TESTS" -m venv "$WORK/venv"
VENV_PYTHON="$WORK/venv/bin/python"
MACHINE="$(uname -m)"
ASSET="$(basename "$WHEEL" | sed -E "s/manylinux_[0-9]+_[0-9]+_[a-z0-9_]+\.whl$/manylinux_2_35_$MACHINE.whl/")"

# kria-apps-firmware with one fake design (the real Makefile, fake bitstream)
mkdir -p "$WORK/firmware-src/kria-apps-firmware-main/boards/kv260/benchmark-b4096"
cp "$HERE/tests/fixtures/kria-apps-firmware-Makefile" "$WORK/firmware-src/kria-apps-firmware-main/boards/Makefile"
printf 'all: { @BIT@ }\n' >"$WORK/firmware-src/kria-apps-firmware-main/boards/template.bif"
design="$WORK/firmware-src/kria-apps-firmware-main/boards/kv260/benchmark-b4096"
echo bitstream >"$design/kv260-benchmark-b4096.bit"
echo "/dts-v1/;" >"$design/kv260-benchmark-b4096.dtsi"
echo xclbin >"$design/kv260-benchmark-b4096.xclbin"
echo '{"shell_type": "XRT_FLAT", "num_slots": "1"}' >"$design/shell.json"
tar -czf "$WORK/firmware.tar.gz" -C "$WORK/firmware-src" kria-apps-firmware-main

fake_kria() {  # fake_kria <dir> <model>
    local root="$1"
    mkdir -p "$root/proc/device-tree" "$root/etc/apt/sources.list.d" "$root/etc/dfx-mgrd" "$root/bin" \
        "$root/lib/firmware/xilinx"
    printf '%s\0' "$2" >"$root/proc/device-tree/model"
    # fake system tools: record the commands; dpkg-query knows the packages in $root/installed,
    # apt-cache the ones in $root/available
    cat >"$root/bin/sudo" <<'S'
#!/usr/bin/env bash
"$@"
S
    for tool in apt-get add-apt-repository usermod; do
        cat >"$root/bin/$tool" <<S
#!/usr/bin/env bash
echo "$tool \$*" >>"\$OPENNVE_VITIS_ROOT/commands"
if [[ "$tool" == apt-get && "\$1" == install ]]; then
    shift
    for package in "\$@"; do
        [[ "\$package" == -* ]] && continue
        echo "\$package" >>"\$OPENNVE_VITIS_ROOT/installed"
        if [[ "\$package" == xlnx-firmware-* ]]; then  # AMD's firmware packages install the design
            name="\${package#xlnx-firmware-}"
            mkdir -p "\$OPENNVE_VITIS_ROOT/lib/firmware/xilinx/\$name"
            echo xclbin >"\$OPENNVE_VITIS_ROOT/lib/firmware/xilinx/\$name/\$name.xclbin"
        fi
    done
fi
S
    done
    cat >"$root/bin/dpkg-query" <<'S'
#!/usr/bin/env bash
package="${@: -1}"
if grep -qx "$package" "$OPENNVE_VITIS_ROOT/installed" 2>/dev/null; then
    [[ "$*" == *'${Version}'* ]] && cat "$OPENNVE_VITIS_ROOT/xrt-version" 2>/dev/null || echo "install ok installed"
else
    exit 1
fi
S
    cat >"$root/bin/apt-cache" <<'S'
#!/usr/bin/env bash
grep -qx "${@: -1}" "$OPENNVE_VITIS_ROOT/available" 2>/dev/null && echo "  Candidate: 1.0" || echo "  Candidate: (none)"
S
    cat >"$root/bin/xmutil" <<'S'
#!/usr/bin/env bash
echo "xmutil $*" >>"$OPENNVE_VITIS_ROOT/commands"
case "$1" in
    boardid) echo "Product Name: SCK-KV-G" ;;
    listapps) cat "$OPENNVE_VITIS_ROOT/apps" 2>/dev/null || true ;;
esac
S
    cat >"$root/bin/bootgen" <<'S'
#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do [[ "$1" == -o ]] && { echo bin >"$2"; }; shift; done
S
    cat >"$root/bin/dtc" <<'S'
#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do [[ "$1" == -o ]] && { echo dtbo >"$2"; }; shift; done
S
    chmod +x "$root/bin/"*
    touch "$root/commands" "$root/installed" "$root/available"
    mkdir -p "$root/assets"
    cp "$WHEEL" "$root/assets/$ASSET"
    cat >"$root/releases.json" <<JSON
[{"tag_name": "v9.9.9", "draft": false, "assets": [
  {"name": "opennve_vitis-0.0.1-cp311-abi3-manylinux_2_35_other.whl", "browser_download_url": "file:///nonexistent"},
  {"name": "$ASSET", "browser_download_url": "file://$root/assets/$ASSET"}]}]
JSON
}

run() {  # run <root> <options...>
    local root="$1"; shift
    OPENNVE_VITIS_RELEASES="file://$root/releases.json" OPENNVE_VITIS_ROOT="$root" \
        OPENNVE_VITIS_FIRMWARE_ARCHIVE="file://$WORK/firmware.tar.gz" CACHE="$root/cache" \
        PYTHON="$VENV_PYTHON" PATH="$root/bin:$PATH" bash "$HERE/install.sh" --yes "$@" >"$root/log" 2>&1 \
        || { cat "$root/log"; echo "install.sh failed"; exit 1; }
}

check() { if ! eval "$1"; then echo "FAILED: $1"; cat "$2/log"; exit 1; fi; echo "ok: $1"; }

# KV260 with nothing set up: PPAs, XRT, design built from kria-apps-firmware, loaded, vart.conf, package
R="$WORK/fresh"; fake_kria "$R" "ZynqMP SMK-K26 Rev1/B/A"
run "$R"
check 'grep -q "add-apt-repository -y ppa:xilinx-apps/ppa" "$R/commands"' "$R"
check 'grep -q "apt-get install -y xrt xrt-dkms" "$R/commands"' "$R"
check '[[ -f "$R/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin" ]]' "$R"
check '[[ -f "$R/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.bin" ]]' "$R"
check '[[ -f "$R/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.dtbo" ]]' "$R"
check 'grep -q "xmutil loadapp kv260-benchmark-b4096" "$R/commands"' "$R"
check 'grep -qx "kv260-benchmark-b4096" "$R/etc/dfx-mgrd/default_firmware"' "$R"
check 'grep -qx "firmware: /lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin" "$R/etc/vart.conf"' "$R"
check 'grep -q "v9.9.9: $ASSET" "$R/log"' "$R"
check 'grep -q "with the Vitis AI Runtime" "$R/log"' "$R"
check '! grep -q "maturin\|build_vart" "$R/log"' "$R"

# everything set up already: nothing installed or changed, the design is not loaded again
R="$WORK/ready"; fake_kria "$R" "ZynqMP SMK-K26 Rev1/B/A"
printf 'xrt\nxrt-dkms\n' >"$R/installed"; echo "202320.2.16.204" >"$R/xrt-version"
echo "deb https://ppa.launchpadcontent.net/xilinx-apps/ppa/ubuntu noble main" >"$R/etc/apt/sources.list.d/xilinx-apps.list"
mkdir -p "$R/lib/firmware/xilinx/kv260-benchmark-b4096"; echo x >"$R/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin"
echo "  kv260-benchmark-b4096   XRT_FLAT  kv260-benchmark-b4096  XRT_FLAT  (0+0)  0," >"$R/apps"
echo "kv260-benchmark-b4096" >"$R/etc/dfx-mgrd/default_firmware"
echo "firmware: /lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin" >"$R/etc/vart.conf"
run "$R"
check '! grep -q "apt-get install\|add-apt-repository\|loadapp" "$R/commands"' "$R"
check 'grep -q "XRT: installed" "$R/log" && grep -q "is installed" "$R/log"' "$R"

# XRT too old (Ubuntu universe 2.8): stops with a clear message
R="$WORK/old"; fake_kria "$R" "ZynqMP SMK-K26 Rev1/B/A"
printf 'xrt\nxrt-dkms\n' >"$R/installed"; echo "2.8.743-0ubuntu11" >"$R/xrt-version"
if OPENNVE_VITIS_RELEASES="file://$R/releases.json" OPENNVE_VITIS_ROOT="$R" PYTHON="$VENV_PYTHON" \
    PATH="$R/bin:$PATH" bash "$HERE/install.sh" --yes >"$R/log" 2>&1; then
    echo "FAILED: old XRT accepted"; cat "$R/log"; exit 1
fi
check 'grep -q "XRT 2.8.743 is too old" "$R/log"' "$R"

# own design (--firmware) from a directory, the apt package of AMD if there is one
R="$WORK/own"; fake_kria "$R" "ZynqMP SMK-K26 Rev1/B/A"
mkdir -p "$R/my-dpu"; echo x >"$R/my-dpu/my-dpu.xclbin"; echo b >"$R/my-dpu/my-dpu.bit.bin"
run "$R" --firmware="$R/my-dpu" --no-boot
check '[[ -f "$R/lib/firmware/xilinx/my-dpu/my-dpu.xclbin" ]]' "$R"
check 'grep -q "xmutil loadapp my-dpu" "$R/commands"' "$R"
check 'grep -qx "firmware: /lib/firmware/xilinx/my-dpu/my-dpu.xclbin" "$R/etc/vart.conf"' "$R"
check '[[ ! -s "$R/etc/dfx-mgrd/default_firmware" ]]' "$R"
R="$WORK/apt"; fake_kria "$R" "ZynqMP SMK-K26 Rev1/B/A"
echo "xlnx-firmware-kv260-benchmark-b4096" >"$R/available"
run "$R"
check 'grep -q "apt-get install -y xlnx-firmware-kv260-benchmark-b4096" "$R/commands"' "$R"

# not a Kria board / --no-system: no system changes, only the package
R="$WORK/pc"; fake_kria "$R" "Some PC"; rm "$R/bin/xmutil"
run "$R"
check '[[ ! -s "$R/commands" ]] && grep -q "no AMD Kria" "$R/log" && grep -q "v9.9.9" "$R/log"' "$R"
R="$WORK/nosys"; fake_kria "$R" "ZynqMP SMK-K26 Rev1/B/A"
run "$R" --no-system
check '! grep -q "apt-get\|loadapp" "$R/commands" && [[ ! -e "$R/etc/vart.conf" ]]' "$R"

# KR260 without own design: AMD has none, clear message
R="$WORK/kr260"; fake_kria "$R" "ZynqMP SMK-K26 Rev1/B/A"
sed -i 's/SCK-KV-G/SCK-KR-G/' "$R/bin/xmutil"
if OPENNVE_VITIS_RELEASES="file://$R/releases.json" OPENNVE_VITIS_ROOT="$R" PYTHON="$VENV_PYTHON" \
    PATH="$R/bin:$PATH" bash "$HERE/install.sh" --yes >"$R/log" 2>&1; then
    echo "FAILED: KR260 without design accepted"; cat "$R/log"; exit 1
fi
check 'grep -q "no ready DPU design for the kr260" "$R/log"' "$R"

echo "all install.sh tests passed"
