"""Checkpoint selection, EMA weight averaging and report-set comparison. No training jobs."""
import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import checkpoints as ck


def write_json(path, value):
    with open(path, "x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with open(path, "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_at(path):
    from fairchem.core.datasets.ase_datasets import AseDBDataset
    return AseDBDataset({"src": str(path), "a2g_args": {"r_edges": False}})


def inventory(path):
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(path.glob("*.aselmdb"))
    if not files:
        raise ValueError("data path needs an ASE database or an OMol25 directory")
    return [{"name": p.name, "bytes": p.stat().st_size} for p in files]


def check_split(rows_path, data):
    """A list named in manifests/*_samples.json indexes one split; refuse a dataset of another size (wrong folder)."""
    digest = ck.sha256(rows_path)
    for manifest in sorted((ROOT / "manifests").glob("*_samples.json")):
        for name, entry in json.loads(manifest.read_text(encoding="utf-8"))["lists"].items():
            if entry["sha256"] == digest and len(data) != entry["rows_in_split"]:
                raise ValueError(f"{name} indexes {entry['split']} ({entry['rows_in_split']} rows); --data has {len(data)} rows")


def release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("select")
    selection.add_argument("--run-dir", type=Path, required=True)
    selection.add_argument("--total-steps", type=int, required=True, help="the run length in settings.txt")
    selection.add_argument("--allow-unfinished", action="store_true",
                           help="a run stopped before --total-steps: use the step checkpoints that exist; outputs are marked partial")
    selection.add_argument("--variant", required=True)
    selection.add_argument("--seed", type=int, required=True)
    selection.add_argument("--k", type=int, default=5)
    selection.add_argument("--selection-rows", type=Path, default=ROOT / "manifests/split_val_select_10k.txt")
    selection.add_argument("--report-rows", type=Path, default=ROOT / "manifests/split_test_eval_10k.txt")
    selection.add_argument("--output", type=Path, required=True, help="new directory")
    average = sub.add_parser("average")
    average.add_argument("--selection", type=Path, required=True)
    average.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--selection", type=Path, required=True)
    report.add_argument("--averaged", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True, help="new directory")
    report.add_argument("--rows", type=Path, help="an extra list fixed before selection instead of the report list, for "
                        "example pilot_val_5k.txt of val (out-of-composition check); needs --eval-split")
    report.add_argument("--eval-split", default="report", help="eval_split column; with --rows a new name, e.g. official_val")
    for command in (selection, report):
        command.add_argument("--data", type=Path, required=True,
                             help="the full, unchanged OMol25 split the lists index: train_4M for split_*, val for the pilot lists")
        command.add_argument("--device", default="cpu")
        command.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    if args.command == "average":
        record = ck.checked_selection(args.selection)
        paths = [r["path"] for r in record["selected"]]
        sidecar = args.output.with_suffix(args.output.suffix + ".json")
        if sidecar.exists():
            raise FileExistsError(sidecar)
        ck.average_weights(paths, args.output)
        write_json(sidecar, {"selection_sha256": ck.sha256(args.selection), "checkpoint_sha256": ck.sha256(args.output),
                             "sources": record["selected"], "weights": "mean EMA parameters", "inference_only": True})
        print(args.output)
        return

    data = dataset_at(args.data)
    if args.command == "select":
        selection_rows, report_rows = ck.split_rows(args.selection_rows, args.report_rows)
        if max(selection_rows + report_rows) >= len(data):
            raise ValueError("row index outside dataset")
        for path in (args.selection_rows, args.report_rows):
            check_split(path, data)
        settings = (args.run_dir / "settings.txt").read_text()
        for name, expected in (("variant", args.variant), ("seed", str(args.seed)), ("steps", str(args.total_steps))):
            match = re.search(rf"(?:^|\s){name}=(\S+)", settings)
            if not match or match.group(1) != expected:
                raise ValueError(f"{name} does not match run settings")
        candidates = ck.discover(args.run_dir, args.total_steps, args.allow_unfinished)
        # An unfinished run applies the same rule to the steps it reached: the last half of 0 to the last checkpoint.
        length = candidates[-1][0]
        partial = dict(partial=length < args.total_steps, reached_step=length) if args.allow_unfinished else {}
        # Fail before expensive inference if K cannot be met.
        if length <= 0 or sum(step >= length / 2 for step, _ in candidates) < args.k or args.k < 1:
            raise ValueError("not enough last-half checkpoints for the requested K")
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
        for step, path in candidates:
            if step < length / 2:
                continue
            unit = ck.predictor(path, args.device, args.dtype)
            row = dict(step=step, path=str(path), sha256=ck.sha256(path), **ck.evaluate(unit, data, selection_rows), **partial)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del unit
            release()
        selected = ck.select(rows, length, args.k)
        record = dict(run_dir=str(args.run_dir.resolve()), variant=args.variant, seed=args.seed,
                      total_steps=args.total_steps, **partial, k=args.k, dtype=args.dtype,
                      rule="lowest selection force MAE in last half; ties use earlier step",
                      selection_rows=str(args.selection_rows.resolve()), report_rows=str(args.report_rows.resolve()),
                      selection_rows_sha256=ck.sha256(args.selection_rows), report_rows_sha256=ck.sha256(args.report_rows),
                      data_inventory=inventory(args.data), data_rows=len(data),
                      selected=selected, final=max(rows, key=lambda r: r["step"]))
        write_csv(args.output / "selection_metrics.csv", rows)
        write_json(args.output / "selection.json", record)
    else:
        record = ck.checked_selection(args.selection)
        if args.dtype != record["dtype"]:
            raise ValueError("use the same precision for selection and reporting")
        for field in ("selection_rows", "report_rows"):
            if ck.sha256(record[field]) != record[field + "_sha256"]:
                raise ValueError("row list changed after selection")
        _, report_rows = ck.split_rows(record["selection_rows"], record["report_rows"])
        same_data = inventory(args.data) == record["data_inventory"] and len(data) == record["data_rows"]
        if (args.rows is None) != (args.eval_split == "report"):
            raise ValueError("--eval-split is report for the report list and a new name (e.g. official_val) with --rows")
        if args.rows:
            # An extra list: never the selection rows; on another dataset (official val) the row numbers do not compare.
            report_rows = ck.read_rows(args.rows)
            if same_data and set(report_rows) & set(ck.read_rows(record["selection_rows"])):
                raise ValueError("extra rows overlap the selection rows")
        elif not same_data:
            raise ValueError("dataset inventory changed after selection")
        if max(report_rows) >= len(data):
            raise ValueError("row index outside dataset")
        check_split(args.rows or record["report_rows"], data)
        sidecar = json.loads(args.averaged.with_suffix(args.averaged.suffix + ".json").read_text())
        if sidecar["selection_sha256"] != ck.sha256(args.selection) or sidecar["checkpoint_sha256"] != ck.sha256(args.averaged):
            raise ValueError("averaged checkpoint does not match this selection")
        partial = {key: record[key] for key in ("partial", "reached_step") if key in record}
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
        methods = {"final": [record["final"]["path"]], "best": [record["selected"][0]["path"]],
                   "weight_average": [str(args.averaged)], "prediction_average": [r["path"] for r in record["selected"]]}
        for method, paths in methods.items():
            units = [ck.predictor(path, args.device, args.dtype) for path in paths]
            unit = ck.PredictionAverage(units)
            row = dict(variant=record["variant"], seed=record["seed"], method=method, eval_split=args.eval_split,
                       **ck.evaluate(unit, data, report_rows), **partial)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del unit, units
            release()
        write_csv(args.output / "ensemble_metrics.csv", rows)
        rows_file = args.rows or Path(record["report_rows"])
        write_json(args.output / "evaluation.json", dict(selection_sha256=ck.sha256(args.selection),
                   averaged_sha256=ck.sha256(args.averaged), dtype=args.dtype, methods=methods, **partial,
                   eval_split=args.eval_split, rows=str(Path(rows_file).resolve()), rows_sha256=ck.sha256(rows_file),
                   data_inventory=inventory(args.data),
                   energy_metric="mean over structures of absolute total energy error divided by atom count",
                   force_metric="mean absolute error over all Cartesian force components"))


if __name__ == "__main__":
    main()
