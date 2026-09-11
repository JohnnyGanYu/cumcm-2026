"""Compatibility CLI/API for the unified Windows runtime packer.

python -m src.build_windows_zip          # sync runtime, manifest and ZIP
python -m src.build_windows_zip --check  # validate without changing files

Runtime selection, newline normalization and deterministic ZIP creation all
belong to pack_dist_windows.sync, so both entry points use the same rules.
"""
from __future__ import annotations

import argparse

if __package__:
    from . import pack_dist_windows as _pack
else:
    # Also support invoking this file by absolute path from another cwd.
    import pack_dist_windows as _pack


def build() -> int:
    """Synchronize the delivery bundle; retain the successful API status 0."""
    _pack.sync()
    return 0


def check() -> int:
    """Return 0 if current, 1 if the ZIP is missing, or 2 for other drift."""
    missing_archive = not _pack.ARCHIVE.is_file()
    stale = _pack.sync(check_only=True)
    if not stale:
        return 0
    return 1 if missing_archive else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true',
                        help='validate only; missing ZIP exits 1, other drift exits 2')
    args = parser.parse_args()
    raise SystemExit(check() if args.check else build())


if __name__ == '__main__':
    main()
