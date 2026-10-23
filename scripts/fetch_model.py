#!/usr/bin/env python3
"""Download the detector weights.

The weights are a 40 MB binary, so they live on a GitHub release rather than in
the repository.  Pass the asset URL, or set KASPAROBOT_MODEL_URL:

    python scripts/fetch_model.py https://github.com/<user>/Kasparobot/releases/download/v1/Kasparobot.pt
"""

import argparse
import os
import pathlib
import sys
import urllib.request

DEST = pathlib.Path(__file__).resolve().parent.parent / "model" / "Kasparobot.pt"
ENV_VAR = "KASPAROBOT_MODEL_URL"


def _progress(blocks, block_size, total):
    if total <= 0:
        return
    done = min(blocks * block_size, total)
    sys.stderr.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
    if done >= total:
        sys.stderr.write("\n")


def fetch(url, dest=DEST):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, partial, _progress)
    partial.replace(dest)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", nargs="?", default=None,
                   help=f"release asset URL (default: ${ENV_VAR})")
    p.add_argument("--dest", default=DEST, type=pathlib.Path,
                   help=f"where to write the weights (default: {DEST})")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    url = args.url or os.environ.get(ENV_VAR)
    if not url:
        print(f"No URL given and ${ENV_VAR} is unset.\n"
              f"Pass a release asset URL, or see model/README.md.")
        return 2
    fetch(url, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
