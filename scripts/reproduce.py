"""Rerun the paper's experiments and assemble the figures and LaTeX tables."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = {
    "fed_batch_reactor": ([8, 16, 32, 64, 128], 10000),
    "ethanol_fermentation": ([32, 64], 1000),
}


def reproduce(run_dir):
    manifest = run_dir / "runs.json"
    runs = json.loads(manifest.read_text()) if manifest.is_file() else {}
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    environment = dict(os.environ, MPLBACKEND="Agg")
    source_dirs = [str(ROOT / "src"), environment.get("PYTHONPATH")]
    environment["PYTHONPATH"] = os.pathsep.join(path for path in source_dirs if path)

    def save_runs():
        pending = manifest.with_suffix(".tmp")
        pending.write_text(json.dumps(runs, indent=2) + "\n")
        pending.replace(manifest)

    def run(log_name, script, *arguments, extra_env=None):
        command = [sys.executable, "-u", str(ROOT / "scripts" / script)]
        command.extend(str(arg) for arg in arguments)
        logfile = logs / f"reproduce-{log_name}.log"
        print(f"Running {log_name} (log: {logfile})", flush=True)
        start = time.monotonic()
        with logfile.open("a") as log:
            log.write(f"$ {shlex.join(command)}\n")
            log.flush()
            try:
                subprocess.run(command, cwd=ROOT, env=environment | (extra_env or {}),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"{log_name} failed. See {logfile}") from error
            log.write(f"\nElapsed: {time.monotonic() - start:.1f} seconds\n")

    def study(example, name, subdirectory, script, *arguments):
        completed = runs.setdefault(example, {})
        if name in completed and (run_dir / completed[name]).is_dir():
            print(f"Using completed {example}/{name}", flush=True)
            return run_dir / completed[name]

        parent = run_dir / example / subdirectory
        before = set(parent.iterdir()) if parent.is_dir() else set()
        run(f"{example}-{name}", f"{example}/{script}.py", *arguments)
        outputs = [path for path in set(parent.iterdir()) - before if path.is_dir()]
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one new results directory for {example}/{name}")
        output = outputs[0]
        completed[name] = str(output.relative_to(run_dir))
        save_runs()
        return output

    save_runs()
    for example, (sizes, replications) in COVERAGE.items():
        study(example, "controls", "controls-states", "nominal_saa")
        clt = study(example, "clt", "limit_theorem", "clt")
        inference = study(example, "inference", "inference", "inference")
        run(f"{example}-reference-plots", "replot_inference.py", "--reference",
            clt / "clt_reference.json", inference)
        worker_args = ["--workers", "auto"] if example == "ethanol_fermentation" else []
        study(example, "coverage", "coverage", "coverage", "--sample-sizes", *sizes,
              "--R", replications, "--n-ref", 4096, *worker_args)

    width = run_dir / "fed_batch_reactor" / "subsampling-width"
    if "width" not in runs or not (run_dir / runs["width"]).is_dir():
        checkpoint = width / "subsampling_width_monte_carlo.json"
        output_args = ["--resume", checkpoint] if checkpoint.is_file() else ["--outdir", width]
        workers = min(8, max(1, (os.cpu_count() or 1) - 2))
        single_threaded = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                               NUMEXPR_NUM_THREADS="1")
        run("fed_batch_reactor-width", "fed_batch_reactor/subsampling_width_monte_carlo.py",
            "--R", 30, "--sample-sizes", 32, 64, "--workers", workers,
            *output_args, extra_env=single_threaded)
        runs["width"] = str(width.relative_to(run_dir))
        save_runs()

    run("prepare-manuscript", "prepare_manuscript.py", "--fresh-run", run_dir)
    print(f"Manuscript figures and tables: {run_dir / 'manuscript'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="continue the replication in results/")
    args = parser.parse_args()
    if shutil.which("pdftocairo") is None:
        parser.error("pdftocairo (Poppler) is required for the EPS exports")

    run_dir = ROOT / "results"
    if args.resume:
        if not (run_dir / "runs.json").is_file():
            parser.error(f"replication manifest not found in {run_dir}")
    elif run_dir.is_dir() and any(run_dir.iterdir()):
        parser.error("Remove or move aside the local results/ directory before a fresh run.")

    run_dir.mkdir(exist_ok=True)
    with (run_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"a replication is already running in {run_dir}")
        try:
            reproduce(run_dir)
        except (RuntimeError, KeyboardInterrupt) as error:
            print(f"{error or 'Replication interrupted'}\n"
                  "Resume with python scripts/reproduce.py --resume", file=sys.stderr)
            raise SystemExit(1) from error


if __name__ == "__main__":
    main()
