"""Regenerate the normal and Student-t coverage tables without SAA solves.

The input stores the optimal value and in-sample standard error for each
replication. The same fixed reference and replications are used for both tables.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm, t


ROOT = Path(__file__).resolve().parents[2]
LEVELS = (0.90, 0.95, 0.99)
DELTA = 1e-6
EXAMPLES = {
    "fed-batch-reactor": ("fed-batch reactor", "table:fed-batch-coverage"),
    "ethanol-fermentation": ("fed-batch ethanol fermentation example", "table:ethanol-coverage"),
}


def coverage_rows(data, method):
    """Return counts, coverage and exact one-sided binomial lower bounds."""
    if method not in ("z", "t"):
        raise ValueError("method must be z or t")
    R = data["R"]
    if not isinstance(R, int) or R < 1:
        raise ValueError("R must be a positive integer")
    reference = float(data["f_ref"])
    if not np.isfinite(reference):
        raise ValueError("reference must be finite")
    rows, seen = [], set()
    for block in data["results"]:
        N = block["N"]
        if not isinstance(N, int) or N < 2 or N in seen:
            raise ValueError("sample sizes must be distinct integers >= 2")
        seen.add(N)
        center = np.asarray(block["J_hat_N"], dtype=float)
        se = np.asarray(block["se"], dtype=float)
        if center.shape != (R,) or se.shape != (R,):
            raise ValueError("each sample size must contain R centers and errors")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(se)):
            raise ValueError("centers and errors must be finite")
        if np.any(se < 0):
            raise ValueError("standard errors must be nonnegative")
        values = []
        for level in LEVELS:
            probability = 1.0 - (1.0 - level) / 2.0
            critical = norm.ppf(probability)
            if method == "t":
                critical = t.ppf(probability, N - 1) * np.sqrt(N / (N - 1))
            halfwidth = critical * se
            count = int(np.count_nonzero(
                (center - halfwidth <= reference)
                & (reference <= center + halfwidth)))
            lower = float(beta.ppf(DELTA, count, R - count + 1)) if count else 0.0
            values.append({"count": count, "coverage": count / R, "lower": lower})
        rows.append({"N": N, "values": values})
    if not rows:
        raise ValueError("at least one sample size is required")
    return sorted(rows, key=lambda row: row["N"])


def latex_tabular(data, method):
    """Render the table body for inclusion in an existing LaTeX float."""
    rows = coverage_rows(data, method)
    lines = [
        r"  \begin{tabular}{rcccccc}", r"    \toprule",
        r"     & \multicolumn{2}{c}{$1-\beta = 0.90$} & \multicolumn{2}{c}{$1-\beta = 0.95$} & \multicolumn{2}{c}{$1-\beta = 0.99$} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"    $N$ & $L/R$ & $\widehat p_{R,10^{-6}}(L)$ & $L/R$ & $\widehat p_{R,10^{-6}}(L)$ & $L/R$ & $\widehat p_{R,10^{-6}}(L)$ \\",
        r"    \midrule",
    ]
    for row in rows:
        cells = [str(row["N"])]
        for value in row["values"]:
            cells.extend([format(value["coverage"], ".3f"), format(value["lower"], ".3f")])
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}"])
    return "\n".join(lines) + "\n"


def latex_table(data, method, example="fed-batch-reactor"):
    """Use the manuscript's three-decimal, six-statistic table layout."""
    name = r"normal plug-in confidence interval" if method == "z" else r"Student-$t$ plug-in confidence interval"
    example_name, label = EXAMPLES[example]
    caption = (
        r"Estimated coverage of the " + name
        + r" for the population optimal value of the " + example_name + ". "
        + r"The population optimum is proxied by the SAA optimal value on an "
        + r"independent reference sample of size $N_{\mathrm{ref}}="
        + str(data["n_ref"]) + r"$. For each training size $N$ and nominal level "
        + r"$1-\beta$, $L/R$ is the empirical coverage over $R=" + str(data["R"])
        + r"$ replications and $\widehat p_{R,10^{-6}}(L)$ is the "
        + r"$(1-10^{-6})$ lower confidence bound on the coverage probability "
        + r"relative to the fixed reference value."
    )
    if method == "t":
        label += "-t"
        caption += (
            r" The interval uses the variance normalization $1/(N-1)$ and "
            r"Student-$t$ critical values with $N-1$ degrees of freedom, "
            r"using the same replications and reference as the normal interval."
        )
    lines = [
        r"\begin{table}[t]", r"  \centering", r"  \caption{" + caption + "}",
        latex_tabular(data, method).rstrip("\n"),
        r"  \label{" + label + "}", r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--example", choices=EXAMPLES, default="fed-batch-reactor")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    revision = ROOT / "results/revision"
    input_name = "coverage_replications.json"
    if args.example == "ethanol-fermentation":
        revision /= "ethanol_coverage"
        input_name = "coverage_plugin_intervals.json"
    input_path = args.input or revision / input_name
    outdir = args.outdir or revision
    with input_path.open() as stream:
        data = json.load(stream)
    # Validate and build both tables before writing either output.
    tables = {name: latex_table(data, method, args.example) for method, name in (
        ("z", "coverage_plugin.tex"), ("t", "coverage_plugin_t.tex"))}
    outdir.mkdir(parents=True, exist_ok=True)
    for name, content in tables.items():
        path = outdir / name
        path.write_text(content)
        print(path)


if __name__ == "__main__":
    main()
