#!/usr/bin/env python3
"""Small source changes to the Vitis AI runtime for openNVE-vitis.

The runtime is built against XRT 2.13 (2022.1), the oldest XRT on Kria Ubuntu,
so the bundled libraries run with every newer XRT the board has as well
(libxrt_coreutil.so.2 keeps its interface). Vitis AI 5.0 calls a few XRT
functions that came later; these are used only where they exist.

Usage: patch_vart.py VAI_RUNTIME_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

COMPAT = """
// openNVE-vitis: xrt::set_read_range exists from XRT 2.15 on. Called through
// argument dependent lookup so that older XRT versions use the fallback.
namespace opennve_xrt_compat {
template <typename Kernel>
auto set_read_range(Kernel& kernel, uint32_t start, uint32_t size, int)
    -> decltype(set_read_range(kernel, start, size), void()) {
  set_read_range(kernel, start, size);
}
template <typename Kernel>
void set_read_range(Kernel&, uint32_t, uint32_t, long) {}
}  // namespace opennve_xrt_compat
"""

PATCHES: list[tuple[str, str, str]] = [
    (
        "vart/xrt-device-handle/src/xrt_device_handle_imp.cpp",
        '#include "vitis/ai/weak.hpp"\n',
        '#include "vitis/ai/weak.hpp"\n' + COMPAT,
    ),
    (
        "vart/xrt-device-handle/src/xrt_device_handle_imp.cpp",
        "xrt::set_read_range(*cu.kernel, 0x10, 0x4F0);",
        "opennve_xrt_compat::set_read_range(*cu.kernel, 0x10, 0x4F0, 0);",
    ),
]


def main(root: Path) -> int:
    for rel, old, new in PATCHES:
        path = root / rel
        text = path.read_text()
        if new in text:
            continue
        if text.count(old) != 1:
            print(f"patch_vart: pattern not found once in {rel}: {old[:60]!r}", file=sys.stderr)
            return 1
        path.write_text(text.replace(old, new))
        print(f"patched {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
