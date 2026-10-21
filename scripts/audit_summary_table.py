"""Compact audit table for the appendix: one row per model and task (12 labels), from results/summary.csv.

Usage: python audit_summary_table.py <results/summary.csv> > table.tex
Columns: Al+Ni mean at 8 and 100 A, the ideal geometry at 100 A, and the verdict counts over the 18 cells
(3 cases x 6 gaps) of the label. Every number is the CSV value rounded for display.
"""
import sys
import pandas as pd

s = pd.read_csv(sys.argv[1])
LABELS = [("medium", "notask", "MACE-MP-0 medium", "none"), ("orb-v3", "notask", "ORB-v3", "none"),
          ("uma-s-1p1", "omat", "UMA-S-1.1", "omat"), ("uma-s-1p1", "omol", "UMA-S-1.1", "omol"),
          ("uma-s-1p2", "omat", "UMA-S-1.2", "omat"), ("uma-s-1p2", "omol", "UMA-S-1.2", "omol"),
          ("uma-m-1p1", "omat", "UMA-M-1.1", "omat"), ("uma-m-1p1", "omol", "UMA-M-1.1", "omol"),
          ("uma-s-1p1-v1", "omat", "UMA-S-1.1 as v1.0", "omat"), ("uma-s-1p1-v1", "omol", "UMA-S-1.1 as v1.0", "omol"),
          ("uma-s-1p2-v1.1", "omat", "UMA-S-1.2 as v1.1", "omat"), ("uma-s-1p2-v1.1", "omol", "UMA-S-1.2 as v1.1", "omol")]


def f(x):
    if x == 0:
        return "$0$"
    if abs(x) < 0.01:
        m, e = f"{x:.1e}".split("e")
        return "$%s\\times10^{%d}$" % (m, int(e))
    if abs(x) >= 1000:
        return "$%s$" % f"{x:+,.1f}".replace(",", "{,}")
    return "$%+.3f$" % x if abs(x) < 1 else "$%+.1f$" % x


rows = []
total = {"VIOLATION": 0, "CONSISTENT": 0, "INCONCLUSIVE": 0}
for m, t, name, task in LABELS:
    d = s[(s.model == m) & (s.task == t)]
    assert len(d) == 18, (m, t, len(d))
    a8 = d[(d.case == "AlNi") & (d.distance == 8)].iloc[0]
    a100 = d[(d.case == "AlNi") & (d.distance == 100)].iloc[0]
    c = d.verdict.value_counts()
    for k in total:
        total[k] += int(c.get(k, 0))
    rows.append(f"{name} & {task} & {int(a100.n)} & {f(a8.mean_mev)} & {f(a100.mean_mev)} & {f(a100.ideal_mev)} & "
                f"{int(c.get('VIOLATION', 0))} / {int(c.get('CONSISTENT', 0))} / {int(c.get('INCONCLUSIVE', 0))} \\\\")
assert total == {"VIOLATION": 110, "CONSISTENT": 64, "INCONCLUSIVE": 42}, total
print("\n".join(rows))
