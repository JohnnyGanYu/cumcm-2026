"""Build/check the standard-library Windows runtime from its tested sources.

python -m src.build_windows
python -m src.build_windows --check
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "dist_windows"
ARCHIVE = ROOT / "CUMCM2026B_robot_windows.zip"
MODULES = ("__init__", "arena", "client", "directional", "geometry", "homing",
           "silence_coverage", "routing", "strategy")


def payloads():
    sources = {f"b_sim/{name}.py": ROOT / "src/b_sim" / f"{name}.py" for name in MODULES}
    sources["run_official.py"] = ROOT / "src/run_official.py"
    sources["README_WINDOWS.md"] = DEST / "README_WINDOWS.md"
    files = {name: path.read_bytes().replace(b"\r\n", b"\n") for name, path in sources.items()}
    files["MANIFEST.sha256"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items())
    ).encode("utf-8")
    return files


def check(files):
    for name, data in files.items():
        path = DEST / name
        if not path.exists() or path.read_bytes().replace(b"\r\n", b"\n") != data:
            raise SystemExit(f"Stale or missing Windows file: {name}")
    if not ARCHIVE.exists():
        raise SystemExit("Windows archive is missing")
    with ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(files):
            raise SystemExit("Windows archive has missing or unexpected members")
        for name, data in files.items():
            if archive.read(name) != data:
                raise SystemExit(f"Stale Windows archive member: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = payloads()
    if not args.check:
        for name, data in files.items():
            target = DEST / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        temporary = ARCHIVE.with_suffix(".zip.tmp")
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(files.items()):
                entry = ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0))
                entry.compress_type = ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                archive.writestr(entry, data)
        temporary.replace(ARCHIVE)
    check(files)
    print(f"Windows directory and ZIP match all {len(files)} manifest members.")


if __name__ == "__main__":
    main()
