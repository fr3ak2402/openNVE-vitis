"""Checks run in a separate process by opennve_vitis (VART stops the process on some errors).

    python -m opennve_vitis._probe cores          {"cores": [{"name", "fingerprint", "batch"}]}
    python -m opennve_vitis._probe check MODEL    {"ok": true, ...} or {"error": "..."}

The last line of the output is JSON. If VART stops the process, the caller reads the reason from
its error output.
"""

from __future__ import annotations

import json
import sys


def _cores() -> list[dict]:
    from . import _opennve_vitis

    return [{"name": name, "fingerprint": fingerprint, "batch": batch}
            for name, fingerprint, batch in _opennve_vitis.dpu_cores()]


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("cores", "check") or (argv[0] == "check" and len(argv) != 2):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        if argv[0] == "cores":
            print(json.dumps({"cores": _cores()}), flush=True)
            return 0
        from . import _opennve_vitis, inspect

        info = inspect(argv[1])
        cores = _cores()
        if not cores:
            print(json.dumps({"error": "the loaded design has no DPU"}), flush=True)
            return 1
        fingerprints = {core["fingerprint"] for core in cores}
        if info.fingerprint and info.fingerprint not in fingerprints:
            loaded = ", ".join(f"{core['name']} (0x{core['fingerprint']:x})" for core in cores)
            print(json.dumps({"error": (
                f"compiled for {info.target or 'another DPU'} (fingerprint 0x{info.fingerprint:x}), "
                f"the loaded design has {loaded}: compile the model with the arch.json of this DPU"
            )}), flush=True)
            return 1
        model = _opennve_vitis._Model(argv[1], "")
        print(json.dumps({"ok": True, "batch": model.batch_size, "target": info.target}), flush=True)
        model.close()
        return 0
    except Exception as error:  # errors VART reports as exceptions
        print(json.dumps({"error": str(error)}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
