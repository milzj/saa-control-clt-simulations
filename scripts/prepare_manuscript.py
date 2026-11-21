"""Collect the paper's EPS figures and LaTeX table bodies from saved results."""

import argparse
import json
from pathlib import Path
from shutil import copyfile, which
import subprocess
from tempfile import TemporaryDirectory

from fed_batch_reactor.coverage_tables import latex_tabular


RESULTS = Path(__file__).resolve().parents[1] / "results"
REVISION = RESULTS / "revision"
FIGURE_EXPORTS = REVISION / "figures"

FIGURES = {
    "figure1a.eps": FIGURE_EXPORTS / "figure1a.eps",
    "figure1b.eps": FIGURE_EXPORTS / "figure1b.eps",
    "figure2a.eps": FIGURE_EXPORTS / "figure2a.eps",
    "figure2b.eps": FIGURE_EXPORTS / "figure2b.eps",
    "figure2c.eps": FIGURE_EXPORTS / "figure2c.eps",
    "figure3a.eps": FIGURE_EXPORTS / "figure3a-reference.eps",
    "figure3b.eps": FIGURE_EXPORTS / "figure3b-reference.eps",
    "figure4a.eps": FIGURE_EXPORTS / "figure4a.eps",
    "figure4b.eps": FIGURE_EXPORTS / "figure4b.eps",
    "figure5a.eps": FIGURE_EXPORTS / "figure5a.eps",
    "figure5b.eps": FIGURE_EXPORTS / "figure5b.eps",
    "figure6a.eps": FIGURE_EXPORTS / "figure6a.eps",
    "figure6b.eps": FIGURE_EXPORTS / "figure6b.eps",
    "figure6c.eps": FIGURE_EXPORTS / "figure6c.eps",
    "figure7a.eps": FIGURE_EXPORTS / "figure7a-reference.eps",
    "figure7b.eps": FIGURE_EXPORTS / "figure7b-reference.eps",
}


def write_manuscript(figures, fed_batch_path, ethanol_path, outdir):
    for source in figures.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    fed_batch = json.loads(fed_batch_path.read_text())
    ethanol = json.loads(ethanol_path.read_text())
    tables = {
        "table1.tex": latex_tabular(fed_batch, "z"),
        "table2.tex": latex_tabular(ethanol, "z"),
        "tableD3.tex": latex_tabular(fed_batch, "t"),
        "tableD4.tex": latex_tabular(ethanol, "t"),
    }

    outdir.mkdir(exist_ok=True)
    for name, source in figures.items():
        copyfile(source, outdir / name)
    for name, contents in tables.items():
        # Suppress the extra space after the tabular when read with \input.
        (outdir / name).write_text(contents.rstrip("\n") + "%\n")
    for name in figures:
        (outdir / name).with_suffix(".pdf").unlink(missing_ok=True)
    print(f"Wrote {len(figures)} EPS figure panels and {len(tables)} tables to {outdir}")


def replot_clt(run_dir, outdir):
    """Render the saved CLT values with a font that embeds minus signs in EPS."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ensemblecontrol import plot_optimization_bias
    from ensemblecontrol.inference_plotting import configure_style
    from saa_clt.clt_plots import plot_clt_normal

    source = str(run_dir / "clt.json")
    configure_style()
    with plt.rc_context({"text.usetex": False, "font.serif": ["DejaVu Serif"]}):
        plot_clt_normal(source, outdir=outdir, formats=("eps",), variants=("theory",))
        plot_optimization_bias(source, outdir=outdir, formats=("eps",))


def prepare_fresh_run(run_root):
    runs = json.loads((run_root / "runs.json").read_text())
    pdftocairo = which("pdftocairo")
    if pdftocairo is None:
        raise RuntimeError("Fresh-run EPS export requires pdftocairo (Poppler).")

    pdfs = {}
    coverage = {}
    with TemporaryDirectory() as temp_dir:
        staging = Path(temp_dir)
        figures = {}
        for example, controls_number, clt_number, inference_number in (
            ("fed_batch_reactor", 1, 2, 3),
            ("ethanol_fermentation", 5, 6, 7),
        ):
            paths = {study: (run_root / name).resolve()
                     for study, name in runs[example].items()}
            controls = paths["controls"]
            inference = paths["inference"]
            for panel, case in (("a", "nominal"), ("b", "saa")):
                pdfs[f"figure{controls_number}{panel}.eps"] = (
                    controls / case / f"{case}_{controls.name}_postprocessed_control.pdf"
                )
            for panel, method in (("a", "plugin"), ("b", "subsampling")):
                pdfs[f"figure{inference_number}{panel}.eps"] = (
                    inference / f"{method}_{inference.name}_ci95_reference.pdf"
                )
            clt = staging / example
            replot_clt(paths["clt"], clt)
            figures.update({
                f"figure{clt_number}a.eps": clt / "clt_clt_N32_theory.eps",
                f"figure{clt_number}b.eps": clt / "clt_clt_N64_theory.eps",
                f"figure{clt_number}c.eps": clt / "optimization_bias.eps",
            })
            coverage[example] = paths["coverage"] / "coverage_plugin_intervals.json"

        width = run_root / runs["width"]
        for panel, size in (("a", 32), ("b", 64)):
            pattern = f"fed-batch-mean-subsampling-width-N{size}-R*.pdf"
            matches = sorted(width.glob(pattern))
            if len(matches) != 1:
                raise ValueError(f"Expected one PDF matching {width / pattern}, found {len(matches)}")
            pdfs[f"figure4{panel}.eps"] = matches[0]

        for name, source in pdfs.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            target = staging / name
            subprocess.run([pdftocairo, "-eps", "-level3", "-r", "600", source, target],
                           check=True)
            figures[name] = target

        write_manuscript(figures, coverage["fed_batch_reactor"],
                         coverage["ethanol_fermentation"], run_root / "manuscript")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-run", type=Path,
                        help="collect the run directories listed in PATH/runs.json",
                        metavar="PATH")
    args = parser.parse_args()

    if args.fresh_run is not None:
        prepare_fresh_run(args.fresh_run)
    else:
        write_manuscript(FIGURES, REVISION / "coverage_replications.json",
                         REVISION / "ethanol_coverage/coverage_plugin_intervals.json",
                         RESULTS / "manuscript")


if __name__ == "__main__":
    main()
