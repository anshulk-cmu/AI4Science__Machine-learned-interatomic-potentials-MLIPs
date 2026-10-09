#!/bin/bash
# Download and unpack the OMol25 train_4M and val splits, then check the archives against the sizes and
# SHA-256 values in manifests/omol25_data_manifest.json. Run it inside a Slurm job.
set -euo pipefail
ROOT=/ocean/projects/che260063p/akumar45
REPO=${1:-$ROOT/repo/AI4Science-MLIPs}
BASE=https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514
mkdir -p "$ROOT/data/omol25" && cd "$ROOT/data/omol25"
for name in train_4M val; do
    curl -L --fail --retry 5 --retry-delay 10 -C - -o "$name.tar.gz" "$BASE/$name.tar.gz"
done
python3 - "$REPO/manifests/omol25_data_manifest.json" <<'PY'
import hashlib, json, os, sys
for archive in json.load(open(sys.argv[1], encoding="utf-8"))["archives"]:
    name = os.path.basename(archive["path"].replace("\\", "/"))
    digest = hashlib.sha256()
    with open(name, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    assert os.path.getsize(name) == archive["size_bytes"], f"{name}: size differs from the manifest"
    assert digest.hexdigest() == archive["sha256"], f"{name}: sha256 differs from the manifest"
    print("archive ok", name, flush=True)
PY
tar -xzf train_4M.tar.gz
tar -xzf val.tar.gz
for split in train_4M val; do
    echo "$split: $(ls "$split"/*.aselmdb | wc -l) shards, $(du -sh "$split" | cut -f1)"
    ls "$split/metadata.npz"
done
