"""The long-run chain (slurm/long_run.sh, slurm/submit_chain.sh) and the Wave 1 job files, with stand-in commands.

Stand-ins replace python, nvidia-smi, sbatch and squeue, so these tests need bash and nothing else (no GPU, no Slurm).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_pilot_data  # noqa: E402
import collect_final_metrics as cm  # noqa: E402
sys.path.insert(0, str(ROOT))
from training.checkpoints import discover  # noqa: E402

BASH = shutil.which("bash")
VARIANTS = ("dense", "global", "local6", "local12", "element", "local_global", "global_frozen", "dense_scaled")
GPU_MIB = {"v100-32": 32768, "l40s-48": 46068, "h100-80": 81559}      # nvidia-smi memory.total on Bridges-2 (pilot summaries)

FAKE_PY = r"""#!/bin/bash
echo "$*" >> "$STUB_CALLS"
shift 2
case "$1" in
    scripts/check_pilot_data.py) echo "PASS  stub list"; echo "RESULT: all checks passed"; exit ${STUB_VERIFY_RC:-0} ;;
    audit/check_checkpoint.py) echo "$2  Delta meV  8 A: 0 [disconnected]"; exit ${STUB_SIZE_RC:-0} ;;
    scripts/check_frozen_router.py) echo "PASS  router frozen"; exit 0 ;;
    -m) ;;
    *) echo "unexpected call $*"; exit 9 ;;
esac
for a in "$@"; do
    case "$a" in +job.timestamp_id=*) NAME=${a#*=} ;; training.steps=*) STEPS=${a#*=} ;; training.ckpt_every=*) CKPT=${a#*=} ;; esac
done
D=$OMOL_RUN_DIR/$NAME
C=$D/checkpoints
[ -e "$D/.distributed-shared-file" ] && echo "stale store file present" >> "$STUB_CALLS"
W=$D/logs/wandb/offline-run-$(date +%s%N)
mkdir -p "$C" "$W" && touch "$W/run.wandb"
last=$(ls -d "$C"/step_*/.metadata 2>/dev/null | sed -E 's#.*/step_([0-9]+)/.*#\1#' | sort -n | tail -1)
save() { mkdir -p "$C/$1" && touch "$C/$1/inference_ckpt.pt" "$C/$1/.metadata"; }
case "$STUB_TRAIN" in
    crash) echo "Traceback: stub crash"; exit 3 ;;
    noprogress) exec sleep 60 ;;
    oom) save step_0; save step_$(( ${last:-0} + CKPT )); kill -KILL $$ ;;
    latekill) save step_0; save step_$(( ${last:-0} + CKPT )); sleep 2; kill -KILL $$ ;;
    killjob) save step_0; save step_$(( ${last:-0} + CKPT )); kill -KILL 0 ;;
    stop) n=$(( ${last:-0} + CKPT )); save step_0; save step_$n
          echo "{'train/loss': 0.2, 'train/step': $(( n - 1 ))}"
          printf 'Finished aggregating metrics: \n  val/omol.val,omol_energy,per_atom_mae: 0.0200\n  val/omol.val,omol_forces,mae: 0.0500\n  val/loss: 0.0800\n'
          echo "{'train/loss': 0.2, 'train/step': $(( n + 2 ))}"
          mkdir -p "$C/preemption_state"; exec sleep 60 ;;
    finish) for (( s=${last:-0}; s<STEPS; s+=CKPT )); do save step_$s; done
            echo "{'train/loss': 0.1, 'train/step': $(( STEPS - 1 ))}"
            printf 'Finished aggregating metrics: \n  val/omol.val,omol_energy,per_atom_mae: 0.0131\n  val/omol.val,omol_forces,mae: 0.0370\n  val/loss: 0.0608\n'
            save final; exit 0 ;;
esac
"""
FAKE_NVIDIA_SMI = """#!/bin/bash
case "$*" in *memory.total*) echo "${STUB_GPU_MIB:-81559}" ;; *memory.used*) echo 1000 ;; *) echo "Stub GPU" ;; esac
"""
FAKE_SBATCH = """#!/bin/bash
echo "$*" >> "$STUB_CALLS"
case "$*" in *--test-only*) echo "sbatch: Job 1 to start at 2026-11-02T00:00:00 using 1 processors"; exit 0 ;; esac
n=$(( $(cat "$STUB_COUNTER" 2>/dev/null || echo 500) + 1 )); echo $n > "$STUB_COUNTER"; echo "$n"
"""
FAKE_SQUEUE = """#!/bin/bash
echo "${STUB_QUEUED:-}"
"""


def posix(path):
    return Path(path).as_posix()


def min_mib(script):
    """variant -> MIN_MIB from the case statement of a shell script."""
    text = (ROOT / "slurm" / script).read_text(encoding="utf-8")
    out = {}
    for names, value in re.findall(r"([\w|]+)\)\s*MIN_MIB=(\d+)", text):
        out.update({name: int(value) for name in names.split("|")})
    return out


@unittest.skipUnless(BASH, "needs bash")
class ShellCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name, text in (("python", FAKE_PY), ("nvidia-smi", FAKE_NVIDIA_SMI), ("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE)):
            path = self.bin / name
            path.write_bytes(text.encode("utf-8"))
            path.chmod(0o755)
        self.calls = self.tmp / "calls.txt"
        self.runs = self.tmp / "runs"

    def run_script(self, script, *args, **env):
        base = {k: v for k, v in os.environ.items() if not k.startswith(("LONG_", "STUB_", "SLURM_"))}
        base.update(PATH=os.pathsep.join([str(self.bin), str(Path(BASH).parent), base["PATH"]]),
                    LONG_SHARED=posix(self.tmp / "shared"), LONG_REPO=posix(ROOT), LONG_RUNS_DIR=posix(self.runs),
                    LONG_PY=posix(self.bin / "python"), STUB_CALLS=posix(self.calls), STUB_COUNTER=posix(self.tmp / "n"))
        base.update({k: str(v) for k, v in env.items()})
        # A new session, so a stand-in that kills its process group (as the scheduler does) cannot reach this test process.
        return subprocess.run([BASH, posix(ROOT / "slurm" / script), *map(str, args)], env=base,
                              capture_output=True, text=True, timeout=180, start_new_session=True)

    def calls_text(self):
        return self.calls.read_text(encoding="utf-8") if self.calls.exists() else ""


class LongRunTests(ShellCase):
    SETTINGS = dict(LONG_STEPS=40, LONG_SEED=0, LONG_EVAL_EVERY=40, LONG_CKPT_EVERY=10, LONG_SEGMENT_HOURS=1)

    def segment(self, job, variant="dense", **env):
        return self.run_script("long_run.sh", variant, **{**self.SETTINGS, "SLURM_JOB_ID": job, **env})

    def folder(self, variant="dense"):
        return self.runs / f"long_{variant}_s0_40"

    def test_refuses_to_start_without_each_required_setting(self):
        for missing in self.SETTINGS:
            with self.subTest(missing=missing):
                env = {k: v for k, v in self.SETTINGS.items() if k != missing}
                r = self.run_script("long_run.sh", "dense", **env)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn(f"set {missing}", r.stdout)
                self.assertFalse(self.runs.exists())
        for bad in (dict(LONG_SEGMENT_HOURS=49), dict(LONG_CKPT_EVERY=41), dict(LONG_STEPS="08"), dict(LONG_SEED=-1)):
            with self.subTest(bad=bad):
                self.assertEqual(self.run_script("long_run.sh", "dense", **{**self.SETTINGS, **bad}).returncode, 2)
        self.assertEqual(self.run_script("long_run.sh", "local24", **self.SETTINGS).returncode, 2)
        self.assertEqual(self.calls_text(), "")

    def test_chain_stops_by_time_resumes_finishes_and_then_exits_at_once(self):
        r = self.segment(101, STUB_TRAIN="stop", LONG_STOP_MARGIN_SEC=3590)
        run = self.folder()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        summary = (run / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("END   train exit 124", summary)
        self.assertIn("RESULT: stopped by time; complete checkpoints up to step 10", summary)
        self.assertFalse((run / "DONE").exists() or (run / "FAILED").exists())
        # Stopped before the planned steps: the readers refuse it unless partial mode is asked for.
        with self.assertRaises(cm.CollectionError):
            cm.collect_run({"path": str(run)}, 40, source="log")
        row, _ = cm.collect_run({"path": str(run)}, 40, source="log", allow_unfinished=True)
        self.assertEqual((row["reached_step"], row["partial"], row["force_mae_eV_per_A"]), (10, True, 0.05))
        with self.assertRaises(ValueError):
            discover(run, 40)
        self.assertEqual([s for s, _ in discover(run, 40, allow_unfinished=True)], [0, 10])
        store = run / "train" / run.name / ".distributed-shared-file"
        store.touch()

        r = self.segment(102, STUB_TRAIN="finish")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((run / "DONE").exists())
        summary = (run / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("END   train exit 0", summary)
        self.assertTrue(summary.rstrip().endswith("RESULT: all stages passed"))
        calls = self.calls_text()
        self.assertNotIn("stale store file present", calls)
        train = [c for c in calls.splitlines() if "-m training.run" in c]
        self.assertEqual(len(train), 2)
        for c in train:
            for arg in (f"+job.timestamp_id={run.name}", "training.steps=40", "training.ckpt_every=10", "training.eval_every=40",
                        "job.seed=0", "rhv:" + posix(ROOT / "manifests/split_train_70.txt"),
                        "rhv:" + posix(ROOT / "manifests/split_val_monitor_5k.txt"),
                        "val_dataset.dataset_configs.omol.splits.val.src=" + posix(self.tmp / "shared/data/omol25/train_4M"),
                        "val.metadata_path=" + posix(self.tmp / "shared/data/omol25/pilot/train_4M_index.npz")):
                self.assertIn(arg, c)
            self.assertNotIn("max_steps", c)
            self.assertNotIn("split_test", c)
        verify = [c for c in calls.splitlines() if "check_pilot_data.py verify" in c]
        self.assertTrue(all("--samples manifests/split_samples.json --samples manifests/long_samples.json "
                            "--samples manifests/pilot_samples.json" in c for c in verify))
        size = [c for c in calls.splitlines() if "check_checkpoint.py" in c]
        self.assertEqual(len(size), 1)
        self.assertIn("final/inference_ckpt.pt --gaps 8 10 15 25 100", size[0])
        segments = (run / "segments.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual([s for s in segments if s.startswith("log ")], ["log logs/train_101.log", "log logs/train_102.log"])
        self.assertEqual(len([s for s in segments if s.startswith("wandb ")]), 2)
        settings = (run / "settings.txt").read_text(encoding="utf-8")
        for key in ("variant", "seed", "steps"):
            self.assertEqual(len(re.findall(rf"(?:^|\s){key}=", settings)), 1)
        self.assertRegex(settings, r"split_train_70\.txt sha256 \w{16}, split_val_monitor_5k\.txt sha256 \w{16} from train_4M ")

        before = self.calls_text()
        r = self.segment(103, STUB_TRAIN="crash")
        self.assertEqual(r.returncode, 0)
        self.assertIn("DONE marker present", r.stdout)
        self.assertEqual(self.calls_text(), before)

        # The readers of the results understand this folder.
        row, provenance = cm.collect_run({"path": str(run)}, 40, source="log")
        self.assertEqual((row["variant"], row["seed"], row["force_mae_eV_per_A"]), ("dense", 0, 0.037))
        self.assertTrue(provenance["source"].endswith("train_102.log"))
        self.assertEqual([s for s, _ in discover(run, 40)], [0, 10, 20, 30, 40])

    def test_lists_come_from_variables(self):
        r = self.segment(151, STUB_TRAIN="finish", LONG_TRAIN_LIST=posix(ROOT / "manifests/long_train_1200k.txt"),
                         LONG_VAL_LIST=posix(ROOT / "manifests/pilot_val_5k.txt"), LONG_VAL_SPLIT="val")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        train = [c for c in self.calls_text().splitlines() if "-m training.run" in c][0]
        for arg in ("long_train_1200k.txt", "pilot_val_5k.txt", "splits.val.src=" + posix(self.tmp / "shared/data/omol25/val"),
                    "val.metadata_path=" + posix(self.tmp / "shared/data/omol25/pilot/val_index.npz")):
            self.assertIn(arg, train)
        self.assertNotIn("split_", train)

    def test_crash_writes_failed_and_stops_the_chain(self):
        r = self.segment(201, STUB_TRAIN="crash")
        self.assertEqual(r.returncode, 1)
        self.assertIn("training exited 3", (self.folder() / "FAILED").read_text(encoding="utf-8"))
        n = len(self.calls_text().splitlines())
        r = self.segment(202, STUB_TRAIN="finish")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(self.calls_text().splitlines()), n)

    def test_no_new_checkpoint_in_a_segment_stops_the_chain(self):
        r = self.segment(301, STUB_TRAIN="noprogress", LONG_STOP_MARGIN_SEC=3590)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("no new checkpoint", (self.folder() / "FAILED").read_text(encoding="utf-8"))

    def test_a_kill_long_before_the_limit_is_a_failure_and_near_it_a_stop(self):
        r = self.segment(311, STUB_TRAIN="oom")                       # limit about 1800 s, killed at once
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("killed (exit 137)", (self.folder() / "FAILED").read_text(encoding="utf-8"))
        segments = (self.folder() / "segments.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(segments[0], "log logs/train_311.log")
        self.assertEqual(len([s for s in segments if s.startswith("wandb ")]), 1)
        (self.folder() / "FAILED").unlink()
        r = self.segment(312, STUB_TRAIN="latekill", LONG_STOP_MARGIN_SEC=3590)   # limit about 9 s
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("RESULT: stopped by time", r.stdout)

    def test_a_segment_killed_with_its_script_keeps_its_log_and_wandb_file(self):
        r = self.segment(321, STUB_TRAIN="killjob")                   # the scheduler kills every process of the job
        self.assertNotEqual(r.returncode, 0)
        run = self.folder()
        self.assertEqual((run / "segments.txt").read_text(encoding="utf-8").splitlines(), ["log logs/train_321.log"])
        self.assertNotIn("END   train", (run / "summary.txt").read_text(encoding="utf-8"))
        r = self.segment(322, STUB_TRAIN="finish")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        segments = (run / "segments.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual([s for s in segments if s.startswith("log ")], ["log logs/train_321.log", "log logs/train_322.log"])
        wandb = [s for s in segments if s.startswith("wandb ")]
        self.assertEqual(len(wandb), 2)
        self.assertEqual(wandb, sorted(wandb))                       # segment order
        self.assertLess(segments.index(wandb[0]), segments.index("log logs/train_322.log"))
        self.assertTrue((run / "DONE").exists())

    def test_changed_settings_and_small_gpu_are_refused(self):
        self.assertEqual(self.segment(401, STUB_TRAIN="crash").returncode, 1)
        (self.folder() / "FAILED").unlink()
        r = self.segment(402, STUB_TRAIN="finish", LONG_EVAL_EVERY=20)
        self.assertEqual(r.returncode, 2)
        self.assertIn("settings differ", r.stdout)
        r = self.segment(403, variant="local12", STUB_TRAIN="finish", STUB_GPU_MIB=GPU_MIB["v100-32"])
        self.assertEqual(r.returncode, 1)
        self.assertIn("too small", (self.folder("local12") / "FAILED").read_text(encoding="utf-8"))

    def test_failed_checks_after_the_final_checkpoint_are_rerun_without_training(self):
        r = self.segment(501, variant="global_frozen", STUB_TRAIN="finish", STUB_SIZE_RC=1)
        run = self.folder("global_frozen")
        self.assertEqual(r.returncode, 1)
        self.assertFalse((run / "DONE").exists() or (run / "FAILED").exists())
        n = self.calls_text().count("-m training.run")
        r = self.segment(502, variant="global_frozen", STUB_TRAIN="crash")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((run / "DONE").exists())
        self.assertEqual(self.calls_text().count("-m training.run"), n)
        self.assertEqual(self.calls_text().count("check_frozen_router.py"), 2)


class SubmitChainTests(ShellCase):
    ARGS = ["--steps", 300000, "--seed", 1, "--segments", 3, "--hours", 47, "--eval-every", 10000, "--ckpt-every", 10000]

    def submit(self, variant="dense", *extra, args=None, **env):
        return self.run_script("submit_chain.sh", variant, *(self.ARGS if args is None else args), *extra, **env)

    def sbatch_calls(self):
        return [c for c in self.calls_text().splitlines()]

    def test_refuses_without_each_required_option(self):
        for i in range(0, len(self.ARGS), 2):
            with self.subTest(missing=self.ARGS[i]):
                r = self.submit(args=self.ARGS[:i] + self.ARGS[i + 2:])
                self.assertEqual(r.returncode, 2)
                self.assertIn(f"{self.ARGS[i]} is required", r.stdout)
        for args in (self.ARGS[:-1], self.ARGS + ["--bogus", 1], [a if a != 47 else 49 for a in self.ARGS]):
            self.assertEqual(self.submit(args=args).returncode, 2)
        self.assertEqual(self.submit("local24").returncode, 2)
        self.assertEqual(self.sbatch_calls(), [])

    def test_chain_uses_afterany_and_the_variant_gpu(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.calls.unlink(missing_ok=True)
                r = self.submit(variant)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                calls = self.sbatch_calls()
                self.assertEqual(len(calls), 3)
                ids = re.findall(r"segment \d of 3: job (\d+)", r.stdout)
                self.assertNotIn("--dependency", calls[0])
                self.assertIn(f"--dependency=afterany:{ids[0]}", calls[1])
                self.assertIn(f"--dependency=afterany:{ids[1]}", calls[2])
                gpu = "h100-80" if variant in ("local6", "local12", "element", "local_global") else "v100-32"
                for c in calls:
                    self.assertIn(f"-A che260063p -p GPU-shared --gpus={gpu}:1 -t 47:00:00 --job-name=long_{variant}_s1_300000", c)
                    for setting in (f"LONG_VARIANT={variant}", "LONG_STEPS=300000", "LONG_SEED=1", "LONG_EVAL_EVERY=10000",
                                    "LONG_CKPT_EVERY=10000", "LONG_SEGMENT_HOURS=47"):
                        self.assertIn(setting, c)
                    self.assertTrue(c.endswith("slurm/long_run.job"))
                    self.assertNotIn("afterok", c)
                self.assertIn(" ".join(ids), (self.runs / f"long_{variant}_s1_300000" / "chains.txt").read_text(encoding="utf-8"))
                self.assertGreaterEqual(GPU_MIB[gpu], min_mib("long_run.sh")[variant])
                self.assertNotIn("LONG_VAL_LIST", calls[0])

    def test_list_variables_reach_the_segments(self):
        r = self.submit("dense", LONG_TRAIN_LIST="/x/a.txt", LONG_VAL_LIST="/x/b.txt", LONG_VAL_SPLIT="val")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("data: train /x/a.txt, validation /x/b.txt from val", r.stdout)
        for c in self.sbatch_calls():
            self.assertIn("LONG_TRAIN_LIST=/x/a.txt,LONG_VAL_LIST=/x/b.txt,LONG_VAL_SPLIT=val", c)

    def test_long_gpu_override_allows_only_l40s_for_the_44_gib_variants(self):
        r = self.submit("element", LONG_GPU="l40s-48")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("on l40s-48: at most 141 SU", r.stdout)
        for c in self.sbatch_calls():
            self.assertIn("--gpus=l40s-48:1", c)
        self.assertGreaterEqual(GPU_MIB["l40s-48"], min_mib("long_run.sh")["element"])
        for variant, gpu in (("local12", "l40s-48"), ("dense", "l40s-48"), ("element", "v100-32"), ("element", "h100-80x")):
            with self.subTest(variant=variant, gpu=gpu):
                self.calls.unlink(missing_ok=True)
                self.assertEqual(self.submit(variant, LONG_GPU=gpu).returncode, 2)
                self.assertEqual(self.sbatch_calls(), [])
        r = self.submit("element", LONG_GPU="h100-80")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_dry_run_submits_nothing(self):
        r = self.submit("local12", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("at most 282 SU", r.stdout)
        calls = self.sbatch_calls()
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("--test-only "))
        self.assertFalse(self.runs.exists())

    def test_refuses_a_second_chain_and_a_failed_run(self):
        r = self.submit(STUB_QUEUED="777")
        self.assertEqual(r.returncode, 1)
        self.assertIn("already queued", r.stdout)
        run = self.runs / "long_dense_s1_300000"
        run.mkdir(parents=True)
        (run / "FAILED").write_text("x", encoding="utf-8")
        self.assertEqual(self.submit().returncode, 1)
        (run / "FAILED").unlink()
        (run / "DONE").write_text("x", encoding="utf-8")
        self.assertEqual(self.submit().returncode, 0)
        self.assertEqual(self.sbatch_calls(), [])


class LaunchFileTests(unittest.TestCase):
    def test_shell_scripts_keep_team_files_group_writable(self):
        for script in ("long_run.sh", "submit_chain.sh", "pilot.sh"):
            self.assertIn("\numask 002", (ROOT / "slurm" / script).read_text(encoding="utf-8"), script)

    def test_gpu_guards_agree_between_pilot_and_long_runs(self):
        self.assertEqual(min_mib("pilot.sh"), min_mib("long_run.sh"))
        self.assertEqual(set(min_mib("long_run.sh")), set(VARIANTS))

    def test_wave1_job_files(self):
        guards = min_mib("pilot.sh")
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                text = (ROOT / "slurm" / f"wave1_{variant}.job").read_text(encoding="utf-8")
                new = variant in ("element", "local12", "local_global")
                gpu = "h100-80" if variant in ("local6", "element", "local12", "local_global") else "v100-32"
                self.assertIn("#SBATCH -A che260063p", text)
                self.assertIn(f"#SBATCH --gpus={gpu}:1", text)
                self.assertGreaterEqual(GPU_MIB[gpu], guards[variant])
                self.assertIn("#SBATCH -t 48:00:00", text)
                self.assertIn(f"#SBATCH --array={'0-4' if new else '3-4'}", text)
                self.assertIn("#SBATCH --mail-type=ALL,TIME_LIMIT_80", text)
                self.assertIn("#SBATCH --mail-user=anshulk@andrew.cmu.edu", text)
                self.assertIn("export PILOT_RUN_PREFIX=final", text)
                self.assertIn("export PILOT_STEPS=${PILOT_STEPS:-50000}", text)
                self.assertNotIn("PILOT_SEED", text)      # the seed is the array task id
                self.assertTrue(text.rstrip().endswith(f'/slurm/pilot.sh" {variant}'))

    @unittest.skipUnless(BASH, "needs bash")
    def test_pilot_overrides_read_val_by_default(self):
        script = 'REPO=/r OMOL_DATA_ROOT=/d; source slurm/pilot_data_overrides.sh; printf "%s\\n" "${DATA_OVERRIDES[@]}"'
        out = subprocess.run([BASH, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        t, v = "train_dataset.dataset_configs.omol.splits.train", "val_dataset.dataset_configs.omol.splits.val"
        self.assertEqual(out, [f"{t}.sample_n=null", f"{t}.no_shuffle=true", f"{t}.metadata_path=/d/pilot/train_4M_index.npz",
                               f"{t}.subset_to=[{{metadata_key:index,op:in,rhv:/r/manifests/pilot_train_200k.txt}}]",
                               f"{v}.src=/d/val", f"{v}.sample_n=null", f"{v}.no_shuffle=true",
                               f"{v}.metadata_path=/d/pilot/val_index.npz",
                               f"{v}.subset_to=[{{metadata_key:index,op:in,rhv:/r/manifests/pilot_val_5k.txt}}]"])

    def test_long_run_job_file(self):
        text = (ROOT / "slurm" / "long_run.job").read_text(encoding="utf-8")
        for line in ("#SBATCH -A che260063p", "#SBATCH --mail-type=ALL,TIME_LIMIT_80", "#SBATCH --mail-user=anshulk@andrew.cmu.edu"):
            self.assertIn(line, text)
        self.assertTrue(text.rstrip().endswith('/slurm/long_run.sh" "${LONG_VARIANT:-}"'))


class SampleManifestTests(unittest.TestCase):
    def test_manifests_combine_and_must_not_overlap(self):
        m = ROOT / "manifests"
        self.assertEqual(set(check_pilot_data.load_samples(None)), {"pilot_train_200k.txt", "pilot_val_5k.txt"})
        both = check_pilot_data.load_samples([m / "long_samples.json", m / "pilot_samples.json"])
        self.assertEqual(set(both), {"long_train_1200k.txt", "select_val_5k.txt", "pilot_train_200k.txt", "pilot_val_5k.txt"})
        with self.assertRaises(ValueError):
            check_pilot_data.load_samples([m / "pilot_samples.json", m / "pilot_samples.json"])

    def test_segment_record_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "segments.txt").write_text("log logs/a.log\nwandb train/x/run.wandb\nlog logs/b.log\n", encoding="utf-8")
            self.assertEqual(cm.segment_record(run), {"logs": ["logs/a.log", "logs/b.log"], "wandb": ["train/x/run.wandb"]})
            (run / "segments.txt").write_text("checkpoint x\n", encoding="utf-8")
            with self.assertRaises(cm.CollectionError):
                cm.segment_record(run)


if __name__ == "__main__":
    unittest.main()
