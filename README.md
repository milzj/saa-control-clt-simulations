[![DOI](https://zenodo.org/badge/1298618702.svg)](https://doi.org/10.5281/zenodo.21365487)
[![arXiv](https://img.shields.io/badge/arXiv-2607.14965-b31b1b.svg)](https://arxiv.org/abs/2607.14965)

# Statistical Inference for Scenario-Based Dynamic Optimization under Uncertainty

Simulation code reproducing the two numerical examples of the manuscript

> Aurya Javeed and Johannes Milz, *Statistical Inference for Scenario-Based Dynamic Optimization under Uncertainty*,
> July 16, 2026, https://arxiv.org/abs/2607.14965

The model equations and numerical settings for each example are documented below.

- [Fed-batch reactor](docs/fed_batch_reactor.md)
- [Fed-batch ethanol fermentation](docs/ethanol_fermentation.md)

## Installation

Use Python 3.14 on Linux or macOS for the versions pinned in
[`requirements.txt`](requirements.txt). The reported runs used an Apple M4
laptop with 16 GB of RAM. The dependencies include EnsembleControl
[v0.0.4](https://doi.org/10.5281/zenodo.21328270) and CasADi with IPOPT.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

A [Dockerfile](docker/Dockerfile) is available for container runs.

## Reproducing the results

Run the commands below from the repository root with the environment active.
The saved results are under [`results/`](results/). Study wrappers write new
runs to timestamped directories under `results/<example>/` and record output
and timings under `logs/<example>/`. Shared settings are in
[`src/saa_clt/config.py`](src/saa_clt/config.py).

### Full replication

In a local checkout, remove or move aside `results/` before starting a fresh
replication. With Poppler's `pdftocairo` installed for EPS export, the following
command reruns both examples, the coverage studies, and the Figure 4 experiment.

```bash
python scripts/reproduce.py
```

The script writes study outputs to `results/`, logs to `logs/`, and the numbered
EPS panels and LaTeX tables to `results/manuscript/`. Allow about 21 hours of
uninterrupted runtime on the Apple M4 setup above. Compare the regenerated plots
with the [archived results on GitHub](https://github.com/milzj/saa-control-clt-simulations/tree/main/results).

To continue an interrupted run, use `--resume` to reuse completed studies and
the Figure 4 checkpoint.

```bash
python scripts/reproduce.py --resume
```

### Manuscript results

The [manuscript results](results/manuscript/) contain the paper's EPS figure
panels and LaTeX coverage table bodies. Regenerate the files from the saved
results with one command.

```bash
python scripts/prepare_manuscript.py
```

Files are named by figure panel or table number, such as `figure3a.eps` and
`tableD4.tex`. Use the EPS files with `\includegraphics` and the table bodies
(`table1.tex`, `table2.tex`, `tableD3.tex`, and `tableD4.tex`) with `\input`
inside the corresponding table floats. For pdfLaTeX, use a TeX installation
with `epstopdf` available for EPS conversion.

### Optimal controls and sampling studies

```bash
for example in fed_batch_reactor ethanol_fermentation; do
  scripts/$example/run_nominal_saa.sh  # Figures 1 and 5
  scripts/$example/run_clt.sh          # Figures 2 and 6
  scripts/$example/run_inference.sh    # Data for Figures 3 and 7
done
```

### Confidence intervals with reference lines (Figures 3 and 7)

Add the reference values from the saved CLT runs to the confidence intervals
from the saved inference runs. These commands use the stored data directly.

```bash
python scripts/replot_inference.py --reference \
  results/fed_batch_reactor/limit_theorem/2026-07-14T15-00-00/clt_reference.json \
  results/fed_batch_reactor/inference/2026-07-12T21-23-21

python scripts/replot_inference.py --reference \
  results/ethanol_fermentation/limit_theorem/2026-07-14T15-16-22/clt_reference.json \
  results/ethanol_fermentation/inference/2026-07-12T23-55-36
```

The `ci95_reference` outputs show the 95% intervals used in the manuscript.
The manuscript panels are available as [PDF and EPS files](results/revision/figures/).

### Subsampling interval widths (Figure 4)

Regenerate the plots from the saved subsampling results for `N=32,64` and
`R=30`.

```bash
scripts/fed_batch_reactor/run_subsampling_width_monte_carlo.sh \
  --resume results/revision/subsampling_width_monte_carlo.json
```

Use a new output directory to rerun the experiment.

```bash
scripts/fed_batch_reactor/run_subsampling_width_monte_carlo.sh \
  --sample-sizes 32 64 --R 30 --workers 8 --outdir results/revision/width-rerun
```

### Coverage tables (Tables 1, 2, D.3, and D.4)

| Example | Sample sizes | Replications per size | Saved results |
| --- | --- | --- | --- |
| Fed-batch reactor | 8, 16, 32, 64, 128 | 10,000 | [`results/revision/`](results/revision/) |
| Ethanol fermentation | 32, 64 | 1,000 | [`results/revision/ethanol_coverage/`](results/revision/ethanol_coverage/) |

Regenerate the normal and Student-`t` coverage tables for both examples
from the saved replications.

```bash
scripts/fed_batch_reactor/run_coverage_tables.sh
scripts/fed_batch_reactor/run_coverage_tables.sh --example ethanol-fermentation
```

The commands write `coverage_plugin.tex` and `coverage_plugin_t.tex` to each
example's results directory. Use `--outdir` to write the tables elsewhere.
Each coverage study uses an independent reference sample of size 4096.

To rerun the ethanol coverage simulations, use the paper's settings.

```bash
scripts/ethanol_fermentation/run_coverage.sh \
  --sample-sizes 32 64 --R 1000 --n-ref 4096 --workers auto
```

To rerun the fed-batch coverage simulations at all five sample sizes, use
the following command.

```bash
scripts/fed_batch_reactor/run_coverage.sh \
  --sample-sizes 8 16 32 64 128 --R 10000 --n-ref 4096
```

For a short coverage run, reduce `--R` and `--n-ref`. The completed ethanol run includes its
[console log](results/revision/ethanol_coverage/coverage.log.gz).

## Tests

```bash
python -m pip install -e ".[test]"
pytest -q
```

## Citation and license

Use the citation in [`CITATION.cff`](CITATION.cff). The code is distributed
under the [MIT License](LICENSE).
