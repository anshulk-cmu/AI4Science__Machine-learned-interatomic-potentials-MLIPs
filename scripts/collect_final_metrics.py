"""Collect final validation metrics without loading models or contacting W&B.

See docs/final-metrics.md. Strict defaults: 8 variants, seeds 0..4, 50,000 steps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
import zlib
from pathlib import Path

VARIANTS = ("dense", "global", "element", "local6", "local12", "local_global", "frozen", "dense_scaled")
METRICS = {
    "val/omol.val,omol_energy,per_atom_mae": "energy_mae_per_atom_eV",
    "val/omol.val,omol_forces,mae": "force_mae_eV_per_A",
    "val/loss": "val_loss",
}
COLUMNS = ["variant", "seed", *METRICS.values()]
KEYS = (*METRICS, "train/step", "val/step", "val/epoch")
FIELD = re.compile(r'''(?<![\w/])["']?(%s)["']?\s*:\s*([^\s,}]+)''' % "|".join(map(re.escape, KEYS)))


class CollectionError(ValueError):
    pass


def variant_name(value):
    value = "frozen" if value == "global_frozen" else value
    if value not in VARIANTS:
        raise CollectionError(f"unknown variant: {value!r}")
    return value


def integer(value, name, minimum=0):
    try:
        result = int(value)
        if isinstance(value, bool) or float(value) != result or result < minimum:
            raise ValueError()
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise CollectionError(f"invalid {name}: {value!r}") from exc


def metric_value(value, name):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise CollectionError(f"invalid {name}: {value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(result) or result < 0:
        raise CollectionError(f"{name} must be finite and nonnegative: {value!r}")
    return result


def metadata(run):
    settings = run / "settings.txt"
    if settings.exists():
        text = settings.read_text(encoding="utf-8-sig")
        fields = {}
        for key in ("variant", "seed", "steps"):
            matches = re.findall(rf"(?:^|\s){key}=([^\s]+)", text)
            if len(set(matches)) != 1:
                raise CollectionError(f"{settings}: missing or conflicting {key}")
            fields[key] = matches[0]
        data = re.search(r"\bdata=(.*?)(?=\s+(?:eval_every|ckpt_every)=|\r?\n|$)", text)
        fields.update(data_id=data.group(1).strip() if data else None, metadata_source=str(settings))
    else:
        configs = sorted(run.rglob("canonical_config.yaml"))
        if len(configs) != 1:
            raise CollectionError(f"{run}: need settings.txt or exactly one canonical_config.yaml")
        import yaml
        cfg = yaml.safe_load(configs[0].read_text(encoding="utf-8-sig"))
        try:
            runner = cfg["runner"]
            backbone = runner["train_eval_unit"]["model"]["backbone"]
            name = backbone["model"].rsplit(".", 1)[-1]
            names = {"eSCNMDBackbone": "dense", "eSCNMDMoeBackbone": "global",
                     "eSCNMDFrozenMoeBackbone": "frozen", "eSCNMDScaledBackbone": "dense_scaled",
                     "eSCNMDLocalGlobalMoeBackbone": "local_global"}
            if name == "eSCNMDLocalMoeBackbone":
                variant = {0: "element", 6: "local6", 12: "local12"}[float(backbone["routing_radius"])]
            else:
                variant = names[name]
            data_id = {kind: runner[kind]["dataset"]["dataset_configs"]
                       for kind in ("train_dataloader", "eval_dataloader") if kind in runner}
            fields = dict(variant=variant, seed=cfg["job"]["seed"], steps=runner["max_steps"],
                          data_id=json.dumps(data_id, sort_keys=True) if data_id else None,
                          metadata_source=str(configs[0]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CollectionError(f"{configs[0]}: unsupported or unresolved canonical config") from exc
    fields["variant"] = variant_name(fields["variant"])
    fields["seed"] = integer(fields["seed"], "seed")
    fields["steps"] = integer(fields["steps"], "target steps", 1)
    summary = run / "summary.txt"
    if summary.exists():
        statuses = re.findall(r"END\s+train\s+exit\s+(\d+)", summary.read_text(encoding="utf-8-sig"))
        if statuses and int(statuses[-1]) != 0:
            raise CollectionError(f"{run}: latest training stage failed (exit {statuses[-1]})")
    return fields


def log_rows(path):
    """One dict per training row or validation block; never join separate evals."""
    block = {}
    with path.open(encoding="utf-8-sig", errors="strict") as stream:
        for line in stream:
            if "Finished aggregating metrics:" in line:
                if block:
                    yield block
                block = {"_validation": True}
            fields = dict(FIELD.findall(line))
            if "train/step" in fields:
                if block:
                    yield block
                    block = {}
                yield {"train/step": fields.pop("train/step")}
            if fields:
                if block.keys() & fields.keys():
                    yield block
                    block = {}
                block.update(fields)
            elif block and line.strip() and "Finished aggregating metrics:" not in line:
                # Other val metrics belong to the same block. Other log messages end it.
                if not line.lstrip().startswith("val/"):
                    yield block
                    block = {}
        if block:
            yield block


def wandb_records(path):
    """Read version-0 .wandb framing with CRC and fragmentation checks, read-only.

    Format reference (W&B's implementation, not the removed DataStore API):
    https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/internal/datastore.py
    """
    with path.open("rb") as stream:
        if stream.read(7) != struct.pack("<4sHB", b":W&B", 0xBEE1, 0):
            raise CollectionError(f"{path}: unsupported W&B file header")
        fragments = None
        while True:
            room = 32768 - stream.tell() % 32768
            if room < 7:
                padding = stream.read(room)
                if padding and padding != b"\0" * room:
                    raise CollectionError(f"{path}: invalid W&B padding")
            header = stream.read(7)
            if not header:
                if fragments is not None:
                    raise CollectionError(f"{path}: truncated W&B record")
                return
            if len(header) != 7:
                raise CollectionError(f"{path}: truncated W&B header")
            checksum, length, kind = struct.unpack("<IHB", header)
            room = 32768 - (stream.tell() - 7) % 32768
            if kind not in (1, 2, 3, 4) or length > room - 7:
                raise CollectionError(f"{path}: invalid W&B record framing")
            payload = stream.read(length)
            if len(payload) != length or zlib.crc32(bytes([kind]) + payload) & 0xFFFFFFFF != checksum:
                raise CollectionError(f"{path}: truncated or corrupt W&B record (CRC mismatch)")
            if kind in (1, 2):
                if fragments is not None:
                    raise CollectionError(f"{path}: unfinished W&B fragmented record")
                if kind == 1:
                    yield payload
                else:
                    fragments = bytearray(payload)
            else:
                if fragments is None:
                    raise CollectionError(f"{path}: W&B fragment without first record")
                fragments.extend(payload)
                if kind == 4:
                    yield bytes(fragments)
                    fragments = None


def history_rows(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected an object")
                    yield row
                except ValueError as exc:
                    raise CollectionError(f"{path}:{number}: invalid history JSON") from exc
        return
    try:
        from wandb.proto.wandb_internal_pb2 import Record
    except ImportError as exc:
        raise CollectionError("reading .wandb files requires wandb and its protobuf dependency") from exc
    for payload in wandb_records(path):
        record = Record()
        try:
            record.ParseFromString(payload)
            if record.HasField("history"):
                row = {}
                for item in record.history.item:
                    key = item.key or "/".join(item.nested_key)
                    if key in KEYS:
                        if key in row:
                            raise ValueError(f"duplicate history key: {key}")
                        row[key] = json.loads(item.value_json)
                yield row
        except Exception as exc:
            raise CollectionError(f"{path}: invalid W&B protobuf/history record: {exc}") from exc


def final_event(paths, reader, steps):
    """Select the last eval of the latest segment, not the best or largest-step eval.

    train/step is zero based. val/step, when present, counts completed optimizer
    steps. W&B _step and val/epoch are not optimizer steps.
    """
    current = None
    final = None
    for path in paths:
        # Explicit segment order; no stale state from a previous attempt is usable.
        current, final = None, None
        for row in reader(path):
            if "train/step" in row:
                current = integer(row["train/step"], "train/step") + 1
                final = None
            if row.get("_validation") or "val/epoch" in row or any(key in row for key in METRICS):
                step = integer(row["val/step"], "val/step") if "val/step" in row else current
                final = (step, row, path)
    if final is None:
        raise CollectionError("latest segment has no validation after its last training step")
    step, event, path = final
    if step != steps or (current is not None and current != steps):
        raise CollectionError(f"{path}: final validation step {step}, last training step {current}; expected {steps}")
    missing = set(METRICS) - event.keys()
    if missing:
        raise CollectionError(f"{path}: incomplete final validation; missing {sorted(missing)}")
    return {column: metric_value(event[key], key) for key, column in METRICS.items()}, str(path)


def source_paths(run, spec, key):
    if key in spec:
        if not isinstance(spec[key], list) or not spec[key]:
            raise CollectionError(f"{run}: {key} must be a nonempty ordered list")
        paths = [(run / p).resolve() for p in spec[key]]
    elif key == "logs":
        paths = [run / "logs" / "train.log"] if (run / "logs" / "train.log").exists() else []
    else:
        paths = sorted({p.resolve() for p in run.rglob("*.wandb")})
        if not paths:
            paths = sorted({p.resolve() for p in run.rglob("wandb-history.jsonl")})
        if len(paths) > 1:
            raise CollectionError(f"{run}: multiple W&B histories; specify their order in --run-manifest")
    if len(set(paths)) != len(paths) or any(not p.is_file() for p in paths):
        raise CollectionError(f"{run}: duplicate or missing {key} source")
    return paths


def collect_run(spec, steps, source="auto"):
    run = Path(spec["path"]).resolve()
    meta = metadata(run)
    if meta["steps"] != steps:
        raise CollectionError(f"{run}: configured for {meta['steps']} steps, requested report is {steps}")
    logs = source_paths(run, spec, "logs") if source != "wandb" else []
    histories = source_paths(run, spec, "wandb") if source != "log" else []
    if source == "wandb" and not histories:
        raise CollectionError(f"{run}: no W&B history")
    if histories:
        metrics, selected = final_event(histories, history_rows, steps)
        precision = "full_precision_history"
        if logs:
            rounded, _ = final_event(logs, log_rows, steps)
            if any(abs(metrics[key] - rounded[key]) > 0.00005001 for key in metrics):
                raise CollectionError(f"{run}: W&B and final train.log metrics disagree beyond rounding")
    elif logs:
        metrics, selected = final_event(logs, log_rows, steps)
        precision = "rounded_train_log"
    else:
        raise CollectionError(f"{run}: no usable metric source")
    row = dict(variant=meta["variant"], seed=meta["seed"], **metrics)
    provenance = dict(**meta, run=str(run), source=selected, precision=precision,
                      source_files=[str(p) for p in histories + logs], step_verified=True)
    return row, provenance


def merge_csv(path, steps):
    path = Path(path).resolve()
    sidecar = path.with_suffix(".provenance.json")
    saved = {}
    if sidecar.exists():
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if info.get("csv_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise CollectionError(f"{sidecar}: provenance does not match the CSV")
        saved = {(r["variant"], r["seed"]): r for r in info["rows"]}
    out = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise CollectionError(f"{path}: expected columns {COLUMNS}")
        for record in reader:
            if set(record) != set(COLUMNS):
                raise CollectionError(f"{path}: malformed CSV row")
            row = dict(variant=variant_name(record["variant"]), seed=integer(record["seed"], "seed"))
            row.update({key: metric_value(record[key], key) for key in METRICS.values()})
            key = (row["variant"], row["seed"])
            if sidecar.exists() and key not in saved:
                raise CollectionError(f"{sidecar}: missing provenance for {key}")
            provenance = dict(saved[key]) if key in saved else dict(
                variant=row["variant"], seed=row["seed"], steps=steps, data_id=None,
                metadata_source=str(path), run=None, source=str(path), source_files=[str(path)],
                precision="legacy_csv_unverified", step_verified=False)
            if provenance["steps"] != steps:
                raise CollectionError(f"{path}: cannot mix {provenance['steps']}-step and {steps}-step reports")
            provenance["merged_from"] = str(path)
            out.append((row, provenance))
    return out


def validate(rows, variants, seeds, allow_partial=False):
    expected = {(v, s) for v in variants for s in range(seeds)}
    seen = set()
    data_ids = set()
    for row, provenance in rows:
        key = (row["variant"], row["seed"])
        if key in seen:
            raise CollectionError(f"duplicate variant/seed {key}; explicitly select one run")
        if key not in expected:
            raise CollectionError(f"unexpected variant/seed {key}")
        seen.add(key)
        if provenance["data_id"]:
            data_ids.add(provenance["data_id"])
    if not seen:
        raise CollectionError("no metrics collected")
    if len(data_ids) > 1:
        raise CollectionError("different training/validation data identifiers; collect separate reports")
    missing = sorted(expected - seen)
    if missing and not allow_partial:
        raise CollectionError(f"missing variant/seed pairs: {missing}; use --allow-partial only for an interim table")
    return missing


def write_outputs(output, rows, missing, variants, seeds, steps):
    output = Path(output).resolve()
    sidecar = output.with_suffix(".provenance.json")
    if output.suffix.lower() != ".csv":
        raise CollectionError("--output must end in .csv")
    protected = {Path(p).resolve() for _, r in rows
                 for p in [r["source"], r["metadata_source"], *r["source_files"], r.get("merged_from")]
                 if p}
    protected |= {p.with_suffix(".provenance.json") for p in list(protected) if p.suffix == ".csv"}
    if output in protected or sidecar in protected:
        raise CollectionError("output must not overwrite an input or its provenance")
    rows = sorted(rows, key=lambda item: (variants.index(item[0]["variant"]), item[0]["seed"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(row for row, _ in rows)
    content = buffer.getvalue().encode("utf-8")
    info = dict(schema_version=1, steps=steps, variants=variants, expected_seeds=list(range(seeds)),
                complete=not missing, missing=missing, csv_sha256=hashlib.sha256(content).hexdigest(),
                rows=[p for _, p in rows])
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for target, data in ((output, content), (sidecar, (json.dumps(info, indent=2) + "\n").encode("utf-8"))):
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
                pending.append((Path(stream.name), target))
                stream.write(data)
        # Both files are prepared before replacing anything. The sidecar hash detects interrupted pairs.
        for temporary, target in pending:
            os.replace(temporary, target)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="explicit run directories (not their parent)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--merge", action="append", default=[], type=Path, help="existing metrics CSV; repeatable")
    parser.add_argument("--run-manifest", type=Path, help="JSON list of runs with explicit ordered logs/wandb segments")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--expected-seeds", type=int, default=5, help="require seeds 0 through N-1")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--source", choices=("auto", "wandb", "log"), default="auto")
    parser.add_argument("--allow-partial", action="store_true", help="permit missing runs, never invalid or duplicate rows")
    args = parser.parse_args(argv)
    try:
        steps = integer(args.steps, "--steps", 1)
        seeds = integer(args.expected_seeds, "--expected-seeds", 1)
        variants = [variant_name(v) for v in args.variants]
        if len(set(variants)) != len(variants):
            raise CollectionError("--variants contains duplicates")
        specs = [{"path": str(p)} for p in args.runs]
        if args.run_manifest:
            if args.output.resolve() == args.run_manifest.resolve():
                raise CollectionError("output must not overwrite the run manifest")
            manifest = json.loads(args.run_manifest.read_text(encoding="utf-8-sig"))
            if not isinstance(manifest, list):
                raise CollectionError("run manifest must be a JSON list")
            for item in manifest:
                if not isinstance(item, dict) or "path" not in item or set(item) - {"path", "logs", "wandb"}:
                    raise CollectionError("manifest entries need path and optional logs/wandb lists")
                specs.append(dict(item, path=str((args.run_manifest.parent / item["path"]).resolve())))
        rows = [item for path in args.merge for item in merge_csv(path, steps)]
        for spec in specs:
            try:
                rows.append(collect_run(spec, steps, args.source))
            except (CollectionError, OSError, ValueError) as exc:
                raise CollectionError(f"{spec['path']}: {exc}") from exc
        missing = validate(rows, variants, seeds, args.allow_partial)
        write_outputs(args.output, rows, missing, variants, seeds, steps)
        rounded = sum(p["precision"] == "rounded_train_log" for _, p in rows)
        unverified = sum(not p["step_verified"] for _, p in rows)
        unknown_data = sum(not p["data_id"] for _, p in rows)
        print(f"Wrote {len(rows)} rows to {args.output}; missing pairs: {len(missing)}")
        print(f"Rounded log rows: {rounded}; legacy rows with unverified steps: {unverified}; unknown data IDs: {unknown_data}")
        return 0
    except (CollectionError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
