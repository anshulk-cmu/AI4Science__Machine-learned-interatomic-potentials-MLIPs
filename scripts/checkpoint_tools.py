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
    selection.add_argument("--total-steps", type=int, required=True, help="actual completed length, not a planned unfinished run")
    selection.add_argument("--variant", required=True)
    selection.add_argument("--seed", type=int, required=True)
    selection.add_argument("--k", type=int, default=5)
    selection.add_argument("--selection-rows", type=Path, default=ROOT / "manifests/select_val_5k.txt")
    selection.add_argument("--report-rows", type=Path, default=ROOT / "manifests/pilot_val_5k.txt")
    selection.add_argument("--output", type=Path, required=True, help="new directory")
    average = sub.add_parser("average")
    average.add_argument("--selection", type=Path, required=True)
    average.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--selection", type=Path, required=True)
    report.add_argument("--averaged", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True, help="new directory")
    for command in (selection, report):
        command.add_argument("--data", type=Path, required=True, help="full, unchanged OMol25 val split in manifest row order")
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
        settings = (args.run_dir / "settings.txt").read_text()
        for name, expected in (("variant", args.variant), ("seed", str(args.seed)), ("steps", str(args.total_steps))):
            match = re.search(rf"(?:^|\s){name}=(\S+)", settings)
            if not match or match.group(1) != expected:
                raise ValueError(f"{name} does not match run settings")
        candidates = ck.discover(args.run_dir, args.total_steps)
        # Fail before expensive inference if K cannot be met.
        if sum(step >= args.total_steps / 2 for step, _ in candidates) < args.k or args.k < 1:
            raise ValueError("not enough last-half checkpoints for the requested K")
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
        for step, path in candidates:
            if step < args.total_steps / 2:
                continue
            unit = ck.predictor(path, args.device, args.dtype)
            row = dict(step=step, path=str(path), sha256=ck.sha256(path), **ck.evaluate(unit, data, selection_rows))
            rows.append(row)
            print(json.dumps(row), flush=True)
            del unit
            release()
        selected = ck.select(rows, args.total_steps, args.k)
        record = dict(run_dir=str(args.run_dir.resolve()), variant=args.variant, seed=args.seed,
                      total_steps=args.total_steps, k=args.k, dtype=args.dtype,
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
        if inventory(args.data) != record["data_inventory"] or len(data) != record["data_rows"]:
            raise ValueError("dataset inventory changed after selection")
        sidecar = json.loads(args.averaged.with_suffix(args.averaged.suffix + ".json").read_text())
        if sidecar["selection_sha256"] != ck.sha256(args.selection) or sidecar["checkpoint_sha256"] != ck.sha256(args.averaged):
            raise ValueError("averaged checkpoint does not match this selection")
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
        methods = {"final": [record["final"]["path"]], "best": [record["selected"][0]["path"]],
                   "weight_average": [str(args.averaged)], "prediction_average": [r["path"] for r in record["selected"]]}
        for method, paths in methods.items():
            units = [ck.predictor(path, args.device, args.dtype) for path in paths]
            unit = ck.PredictionAverage(units)
            row = dict(variant=record["variant"], seed=record["seed"], method=method, eval_split="report",
                       **ck.evaluate(unit, data, report_rows))
            rows.append(row)
            print(json.dumps(row), flush=True)
            del unit, units
            release()
        write_csv(args.output / "ensemble_metrics.csv", rows)
        write_json(args.output / "evaluation.json", dict(selection_sha256=ck.sha256(args.selection),
                   averaged_sha256=ck.sha256(args.averaged), dtype=args.dtype, methods=methods,
                   energy_metric="mean over structures of absolute total energy error divided by atom count",
                   force_metric="mean absolute error over all Cartesian force components"))


if __name__ == "__main__":
    main()
