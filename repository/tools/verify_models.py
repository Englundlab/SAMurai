"""Check local model files against the release manifest; never downloads files."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=root / "model")
    parser.add_argument("--all", action="store_true", help="Also require the alternative YOLO model")
    args = parser.parse_args()
    manifest = json.loads((root / "model" / "manifest.json").read_text())
    failed = False
    for item in manifest["models"]:
        if not item["default"] and not args.all:
            continue
        path = args.model_dir / item["path"]
        if not path.is_file():
            print("MISSING:", item["path"]); failed = True; continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        valid = path.stat().st_size == item["bytes"] and digest.hexdigest() == item["sha256"]
        print("OK:" if valid else "MISMATCH:", item["path"])
        failed |= not valid
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
