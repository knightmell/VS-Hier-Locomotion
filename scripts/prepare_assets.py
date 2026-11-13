"""Copy publisher-provided locomotion assets after verifying their hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Downloaded publisher folder (the dppo/log layout)")
    parser.add_argument("--destination", type=Path, default=ROOT / "assets/pretrained")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.source is None:
        parser.error("provide --source, or use --check for existing assets")
    manifest = json.loads((ROOT / "assets_manifest.json").read_text())
    for item in manifest["files"]:
        destination = args.destination / item["destination"]
        source = destination if args.check else args.source / item["upstream_layout"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Asset hash mismatch: " + str(source))
        if not args.check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
        print("verified", item["destination"])


if __name__ == "__main__":
    main()
