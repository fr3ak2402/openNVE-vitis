#!/usr/bin/env python3
"""Copies the Vitis AI Runtime into a wheel built by maturin (like auditwheel repair).

    repair_wheel.py WHEEL PREFIX OUTDIR [--plat manylinux_2_35_aarch64]

* opennve_vitis/lib/ gets the runtime libraries the module needs, the VART plugins VART loads by
  name (DPU runner, async runner, dummy runner, DPU controller) and what they need (abseil,
  protobuf, glog, boost, ...), found with ldd. Not copied: XRT (libxrt_core / libxrt_coreutil, they
  belong to the zocl driver of the board) and the libraries every Linux has (glibc, libstdc++).
* Libraries that are not part of VART get a unique name (libprotobuf-onv.so.28.3.0), so they can
  not collide with other versions in the same process.
* Every library finds the others through RUNPATH $ORIGIN, the module through $ORIGIN/lib.
* The licenses of the bundled libraries go to opennve_vitis/lib/licenses.

Needs patchelf.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

PACKAGE = "opennve_vitis"
# loaded by VART with dlopen("<name>"): they keep exactly this name
PLUGINS = {
    "libvart-dpu-runner.so": "libvart-dpu-runner.so",
    "libvart-async-runner.so": "libvart-async-runner.so",
    "libvart-dummy-runner.so": "libvart-dummy-runner.so",
}
# loaded by opennve_vitis itself with dlopen (dpu_cores): versioned name
EXTRA = ["libvart-dpu-controller.so.5"]
# part of every Linux (glibc, libstdc++; zlib and libuuid are required packages of Ubuntu) or of
# the board's XRT installation
SYSTEM = re.compile(
    r"^(linux-vdso|ld-linux|libc\.so|libm\.so|libdl\.so|librt\.so|libpthread\.so|libstdc\+\+\.so|libgcc_s\.so|"
    r"libresolv\.so|libutil\.so|libnsl\.so|libz\.so|libuuid\.so|libxrt_core\.so|libxrt_coreutil\.so)"
)
VART = re.compile(r"^lib(vart-|xir\.|unilog\.|target-factory\.)")


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def needed(path: pathlib.Path) -> list[str]:
    return [line.split("[", 1)[1].split("]", 1)[0]
            for line in run("readelf", "-d", str(path)).splitlines() if "(NEEDED)" in line]


def resolve(roots: list[pathlib.Path], libdir: pathlib.Path) -> dict[str, pathlib.Path]:
    """soname -> file for everything the roots need (ldd, with the runtime's lib directory first)."""
    env = {**os.environ, "LD_LIBRARY_PATH": f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    found: dict[str, pathlib.Path] = {}
    for root in roots:
        output = subprocess.run(["ldd", str(root)], capture_output=True, text=True, env=env).stdout
        for line in output.splitlines():
            match = re.match(r"\s*(\S+) => (\S+)", line)
            if not match:
                continue
            name, target = match.groups()
            if target == "not" or SYSTEM.match(name):
                if target == "not" and not SYSTEM.match(name):
                    sys.exit(f"repair_wheel: {root.name} needs {name}, not found")
                continue
            found[name] = pathlib.Path(target).resolve()
    return found


def new_name(soname: str) -> str:
    if soname in PLUGINS:
        return PLUGINS[soname]
    if VART.match(soname):
        return soname
    base, _, rest = soname.partition(".so")
    return f"{base}-onv.so{rest}"


def digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wheel", type=pathlib.Path)
    parser.add_argument("prefix", type=pathlib.Path, help="prefix of ci/build_vart.sh")
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--plat", help="platform tag, for instance manylinux_2_35_aarch64 (default: keep)")
    args = parser.parse_args()
    libdir = (args.prefix / "lib").resolve()

    with tempfile.TemporaryDirectory() as temp:
        tree = pathlib.Path(temp)
        with zipfile.ZipFile(args.wheel) as wheel:
            wheel.extractall(tree)
        package = tree / PACKAGE
        modules = sorted(package.glob("_opennve_vitis*.so"))
        if len(modules) != 1:
            sys.exit(f"repair_wheel: expected one extension module in {args.wheel}, found {modules}")
        module = modules[0]

        roots = [module] + [libdir / name for name in [*(f"{p}.5" for p in PLUGINS), *EXTRA]]
        for root in roots:
            if not root.exists():
                sys.exit(f"repair_wheel: {root} missing")
        libraries = resolve(roots, libdir)
        for name in [p.name for p in roots[1:]]:
            libraries.setdefault(name, (libdir / name).resolve())
        # plugins: "libvart-dpu-runner.so.5" -> "libvart-dpu-runner.so"
        renames = {}
        for soname in libraries:
            plain = soname.split(".so")[0] + ".so"
            renames[soname] = PLUGINS.get(plain) or new_name(soname)

        target = package / "lib"
        target.mkdir()
        for soname, source in sorted(libraries.items()):
            destination = target / renames[soname]
            shutil.copy2(source, destination)
            destination.chmod(0o755)
        for soname, name in renames.items():
            path = target / name
            run("patchelf", "--set-soname", name, str(path))
            run("patchelf", "--set-rpath", "$ORIGIN", str(path))
        for path in [module, *(target / name for name in renames.values())]:
            replace = []
            for dependency in needed(path):
                if dependency in renames and renames[dependency] != dependency:
                    replace += ["--replace-needed", dependency, renames[dependency]]
            if replace:
                run("patchelf", *replace, str(path))
        run("patchelf", "--set-rpath", "$ORIGIN/lib", str(module))

        licenses = args.prefix / "share" / "licenses" / "opennve-vitis"
        (target / "licenses").mkdir()
        for path in sorted(licenses.glob("*")):
            shutil.copy2(path, target / "licenses" / path.name)
        if any(name.startswith("libboost") for name in libraries):
            (target / "licenses" / "boost.txt").write_text(BOOST_LICENSE)

        # check: every bundled library loads its dependencies from the bundle
        env = {**os.environ, "LD_LIBRARY_PATH": ""}
        for path in [module, *sorted(target.glob("*.so*"))]:
            output = subprocess.run(["ldd", str(path)], capture_output=True, text=True, env=env).stdout
            missing = [line.strip() for line in output.splitlines()
                       if "not found" in line and not SYSTEM.match(line.strip())]
            if missing:
                sys.exit(f"repair_wheel: {path.name}: {missing}")

        # dist-info: platform tag and RECORD
        dist_info = next(tree.glob("*.dist-info"))
        name = args.wheel.name
        if args.plat:
            old_plat = name[:-4].split("-")[-1]
            name = name.replace(old_plat, args.plat)
            wheel_file = dist_info / "WHEEL"
            wheel_file.write_text(wheel_file.read_text().replace(old_plat, args.plat))
        record = dist_info / "RECORD"
        rows = []
        for path in sorted(p for p in tree.rglob("*") if p.is_file() and p != record):
            data = path.read_bytes()
            rows.append([path.relative_to(tree).as_posix(), digest(data), str(len(data))])
        rows.append([record.relative_to(tree).as_posix(), "", ""])
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        record.write_text(buffer.getvalue())

        args.outdir.mkdir(parents=True, exist_ok=True)
        output = args.outdir / name
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as wheel:
            files = sorted(p for p in tree.rglob("*") if p.is_file())
            # dist-info last, as in wheels from build tools
            for path in sorted(files, key=lambda p: (p.relative_to(tree).parts[0].endswith(".dist-info"), str(p))):
                info = zipfile.ZipInfo(path.relative_to(tree).as_posix())
                info.external_attr = (path.stat().st_mode & 0o777 | 0o100000) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                wheel.writestr(info, path.read_bytes())
        size = output.stat().st_size / 1e6
        print(f"{output} ({len(libraries)} libraries, {size:.1f} MB)")
    return 0


BOOST_LICENSE = """Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless they are solely in the form
of machine-executable object code generated by a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

if __name__ == "__main__":
    sys.exit(main())
