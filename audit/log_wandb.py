"""Log the finished audit to Weights and Biases: the tables, the plots, and the raw result files as an artifact."""
from pathlib import Path

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    run = wandb.init(project="mlip-audit", name="audit-baseline-2026-10-04", job_type="audit",
                     config={"gaps_angstrom": [8, 10, 15, 25, 50, 100], "runs_per_cell": "1 ideal + 20 shaken",
                             "shake_angstrom": 0.05, "cutoff_angstrom": 6, "slab": "Al(111) and Ni(111), 8 layers, 4x4"})
    for name in ("summary", "flatness", "comparison"):
        run.log({name: wandb.Table(dataframe=pd.read_csv(RES / f"{name}.csv"))})
    run.log({p.stem: wandb.Image(str(p)) for p in sorted((RES / "figures").glob("*.png"))})
    art = wandb.Artifact("audit-results", type="results")
    art.add_dir(str(RES))
    art.add_file(str(ROOT / "status-wip" / "README.md"), name="report/README.md")
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
