# saa-control-clt-simulations

Simulation code reproducing the two numerical examples of the manuscript
*Statistical Limit Theorems for Optimal Control under Uncertainty* (Javeed and Milz):

- a **fed-batch reactor** under parametric uncertainty ([docs](docs/fed_batch_reactor.md)), and
- a **fed-batch ethanol fermentation** reactor under parametric uncertainty ([docs](docs/ethanol_fermentation.md)).

For each example the code computes, for the sample-average-approximation (SAA) optimal
value: the nominal and risk-neutral (SAA) solutions, a central-limit-theorem study, the
plug-in and subsampling confidence intervals, and a plug-in coverage study.

The numerical routines come from the external package
[EnsembleControl](https://github.com/milzj/EnsembleControl); this repository pins the
exact release **v0.0.4** archived on Zenodo
([doi:10.5281/zenodo.21328270](https://doi.org/10.5281/zenodo.21328270)), which
`pip install` resolves automatically from the pinned archive URL in
[`pyproject.toml`](pyproject.toml).

## Requirements

- **Operating system:** Linux or macOS.
- **Python:** 3.10 or newer.
- **Hardware:** the results were produced on a laptop with an **Apple M4 chip and
  16 GB of RAM**. No GPU is required.
- All Python dependencies (`numpy`, `scipy`, `casadi`, `matplotlib`, `pandas`, and
  `ensemblecontrol`) are declared in `pyproject.toml` and installed by the commands
  below. IPOPT ships inside the `casadi` wheel; no separate solver install is needed.

## Installation

Create a virtual environment inside the repository and install the package (which
pulls in every dependency, including the pinned EnsembleControl archive):

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install .                        # add [test] for the smoke tests: pip install -e ".[test]"
```

### Using [uv](https://docs.astral.sh/uv/)

The same works with uv:

```bash
uv venv
source .venv/bin/activate
uv pip install .                     # or: uv pip install -e ".[test]"
```

The run scripts prefer the repository's `.venv/bin/python`, so once `.venv` exists you
can run the studies without activating it; activating it also works.

## Reproducing the results

Each example exposes the four studies as separate, independently runnable Python files
under `scripts/<example>/`, each wrapped by a shell script that times it and captures
stdout **and** stderr to a log file:

```bash
# Run everything for one example (all four studies, in order):
scripts/fed_batch_reactor/run_all.sh
scripts/ethanol_fermentation/run_all.sh

# ... or run a single study on its own, e.g.:
scripts/fed_batch_reactor/run_clt.sh
scripts/ethanol_fermentation/run_inference.sh
```

The four studies, and the manuscript figure/table each produces, are:

| Script (`run_*.sh` / `*.py`) | Study | Produces |
| --- | --- | --- |
| `nominal_saa` | Nominal + risk-neutral (SAA) solution | postprocessed control/state figures |
| `clt`         | Central-limit-theorem study | scaled-error histograms + optimization-bias figure |
| `inference`   | Plug-in (Algorithm 1) + subsampling (Algorithm 2) CIs | confidence-interval figures |
| `coverage`    | Plug-in coverage study | coverage LaTeX table (`coverage_plugin.tex`) |

> **Warning — the coverage study is slow.** `run_coverage.sh` (and hence `run_all.sh`)
> solves hundreds of SAA problems per sample size and can take a long time. Use the
> `--R`/`--n-ref` flags for a cheaper smoke run, e.g.
> `scripts/fed_batch_reactor/run_coverage.sh --R 20 --n-ref 128`.

Outputs are written to (both are git-ignored and regenerated on each run):

- `results/<example>/<study>/<stamp>/` — figures, CSVs, JSON, and LaTeX tables;
- `logs/<example>/<study>_<stamp>.log` — the captured console output and timing.

The default sample sizes, seeds, tolerances, and mesh sizes are fixed in
[`src/saa_clt/config.py`](src/saa_clt/config.py) and reproduce the settings used for the
manuscript.

## Repository layout

```
src/saa_clt/     shared, de-duplicated helpers (run configuration, warm starts,
                 control post-processing, output paths)
scripts/<ex>/    each example: model.py, config.py, the four study drivers, run_*.sh
docs/            per-example problem formulations
test/            fast pytest smoke suite
results/ logs/   generated on run (git-ignored)
```

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

The smoke suite exercises the whole pipeline on tiny problem sizes (it checks that the
models simulate and each study runs end-to-end and writes well-formed output); it does
not assert the full-precision numerical results.

## Docker

A container image with the code and all dependencies preinstalled is published to the
GitHub Container Registry on each version tag (see `docker/Dockerfile` and the
`Create and publish a Docker image` workflow):

```bash
docker pull ghcr.io/milzj/saa-control-clt-simulations:latest
docker run --rm -it ghcr.io/milzj/saa-control-clt-simulations
# inside the container, e.g.:
scripts/fed_batch_reactor/run_clt.sh
```

## Citation

If you use this code, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).
