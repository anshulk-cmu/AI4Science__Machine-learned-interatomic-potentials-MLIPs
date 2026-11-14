"""Table of the size-consistency gap from `audit/check_checkpoint.py --runs N` outputs, one text file per variant.

  python scripts/size_consistency_table.py DIR --out results/final_size_consistency.csv
DIR holds <variant>.txt. Writes one row per variant, run and gap, then prints mean, standard deviation, min and max
over the runs for each variant and gap (meV).
"""
import argparse
import csv
import re
import statistics
from pathlib import Path

LINE = re.compile(r"run (\d+)\s+Delta meV\s+(.*)$")
GAP = re.compile(r"(\d+) A: (\S+) \[([a-z-]+)\]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for path in sorted(a.dir.glob("*.txt")):
        text = path.read_text()
        if "FAILED" in text:
            raise ValueError(f"{path} holds a failed checkpoint")
        for m in LINE.finditer(text):
            for gap, value, status in GAP.findall(m.group(2)):
                rows.append(dict(variant=path.stem, run=int(m.group(1)), gap_A=int(gap), delta_meV=float(value), status=status))
    with open(a.out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("variant,gap_A,runs,mean_meV,sd_meV,min_meV,max_meV,status")
    for key in sorted({(r["variant"], r["gap_A"]) for r in rows}):
        own = [r for r in rows if (r["variant"], r["gap_A"]) == key]
        v = [r["delta_meV"] for r in own]
        print(f"{key[0]},{key[1]},{len(v)},{statistics.mean(v):.6g},{statistics.stdev(v):.3g},{min(v):.6g},{max(v):.6g},"
              f"{'/'.join(sorted({r['status'] for r in own}))}")


if __name__ == "__main__":
    main()
