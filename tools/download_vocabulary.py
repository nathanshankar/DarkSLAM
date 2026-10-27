#!/usr/bin/env python3
"""Download the public SuperPoint vocabulary without Git LFS or Python dependencies."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


SHA256 = "f07f5c0fee7d1753f2464354d2834397b722cb0574636cff1420f50f3984498a"
URL = "https://github.com/nathanshankar/DarkSLAM/releases/download/inference-assets/superpoint_voc.dbow3"
DEFAULT = Path(__file__).resolve().parents[1] / "Vocabulary/superpoint_voc.dbow3"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination):
    destination = Path(destination)
    if destination.is_file() and checksum(destination) == SHA256:
        print(f"Vocabulary already present and verified: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(URL, headers={"User-Agent": "DarkSLAM-asset-downloader"})
        with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".vocabulary-", suffix=".part", delete=False
        ) as stream:
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            total = 0
            next_report = 16 * 1024 * 1024
            print(f"Downloading SuperPoint vocabulary to {destination}", flush=True)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total >= next_report:
                    print(f"  Downloaded {total / (1024 * 1024):.0f} MiB", flush=True)
                    next_report += 16 * 1024 * 1024
        if digest.hexdigest() != SHA256:
            raise ValueError("Vocabulary SHA-256 mismatch; existing destination was preserved")
        os.replace(temporary, destination)
        print(f"Vocabulary downloaded and SHA-256 verified: {destination}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        download(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Vocabulary download failed: {error}\n")


if __name__ == "__main__":
    main()
