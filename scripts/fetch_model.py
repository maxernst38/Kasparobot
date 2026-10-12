#!/usr/bin/env python3
"""Download the detector weights.

The weights are a 40 MB binary, so they live on a GitHub release rather than in
the repository.  Pass the asset URL, or set KASPAROBOT_MODEL_URL:

    python scripts/fetch_model.py https://github.com/<user>/Kasparobot/releases/download/v1/Kasparobot.pt
"""

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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    url = argv[0] if argv else os.environ.get(ENV_VAR)
    if not url:
        print(__doc__.strip())
        print(f"\nNo URL given and ${ENV_VAR} is unset.")
        return 2
    fetch(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
