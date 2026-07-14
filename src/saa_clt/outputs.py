"""Output-path helper: route every driver's results to a single top-level
``results/<example>/<study>/<stamp>/`` tree at the repository root.

Each driver lives in ``scripts/<example>/`` and calls
``study_dir(__file__, "<study>", stamp)``; the example name is taken from the
driver's parent folder and the repository root is located by walking up to the
directory that contains ``pyproject.toml``.
"""

import json
import os

import numpy as np
from scipy.stats import norm

__all__ = ["repo_root", "example_name", "study_dir", "latest_run_dir",
           "save_coverage_intervals", "load_coverage_intervals"]


def repo_root(start):
    """Nearest ancestor directory of ``start`` that contains ``pyproject.toml``."""
    d = os.path.abspath(os.path.dirname(start))
    while True:
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                "could not locate the repository root (no pyproject.toml above "
                "%s)" % start)
        d = parent


def example_name(caller_file):
    """The example slug = the driver's parent folder name (e.g. fed_batch_reactor)."""
    return os.path.basename(os.path.dirname(os.path.abspath(caller_file)))


def study_dir(caller_file, study, stamp, make=True):
    """``<repo>/results/<example>/<study>/<stamp>`` for the calling driver.

    ``caller_file`` is the driver's ``__file__``; ``study`` is the study
    sub-folder (``controls-states`` / ``limit_theorem`` / ``inference`` /
    ``coverage``); ``stamp`` is the run timestamp.  Creates the directory when
    ``make`` (default).
    """
    path = os.path.join(repo_root(caller_file), "results",
                        example_name(caller_file), study, stamp)
    if make:
        os.makedirs(path, exist_ok=True)
    return path


def latest_run_dir(caller_file, study, contains=None):
    """Newest ``<repo>/results/<example>/<study>/<stamp>/`` folder for the caller.

    Stamps are ``%Y-%m-%dT%H-%M-%S``, so the lexicographic max is the newest run.
    When ``contains`` is given, only folders holding a file of that name are
    considered -- e.g. ``contains="coverage_plugin_intervals.json"`` skips coverage
    runs made before the endpoint writer existed, and any run that died before
    writing it.  Returns ``None`` when no folder matches; creates nothing.
    """
    root = os.path.join(repo_root(caller_file), "results",
                        example_name(caller_file), study)
    if not os.path.isdir(root):
        return None
    stamps = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if contains is not None and not os.path.exists(os.path.join(path, contains)):
            continue
        stamps.append(name)
    return os.path.join(root, max(stamps)) if stamps else None


def save_coverage_intervals(study, json_path, meta=None):
    """Persist the RAW per-replication plug-in CI endpoints from a coverage study.

    ``ensemblecontrol.coverage_study`` returns the interval endpoints in
    ``study["bounds_by_N"][N][level] = {"lo": array[R], "hi": array[R]}`` but
    ``ensemblecontrol.save_coverage_run`` keeps only the cover/no-cover indicators.
    This writes those endpoints, for every one of the R replications, to ``json_path``
    -- the companion to ``coverage_plugin.json`` -- so interval widths, miss distances,
    and the per-replicate SAA optimum can be inspected without re-running the study.

    The plug-in CI is symmetric about J_hat_N* (``lo = Jhat - z*se``, ``hi = Jhat +
    z*se``), so the center ``J_hat_N = (lo+hi)/2`` (the per-replicate SAA optimal value)
    and the level-independent standard error ``se = (hi-lo)/(2z)`` are recovered from the
    endpoints and stored alongside them.  Per-level arrays are lists aligned with
    ``levels`` (matching ``save_coverage_run``, which avoids float JSON keys).  Pure; does
    no solving.  Returns ``json_path``.
    """
    levels = list(study["levels"])
    z = [float(norm.ppf(1.0 - (1.0 - lvl) / 2.0)) for lvl in levels]
    bounds_by_N = study["bounds_by_N"]

    results = []
    for N in sorted(bounds_by_N):
        per = bounds_by_N[N]
        lo = [np.asarray(per[lvl]["lo"], dtype=float) for lvl in levels]
        hi = [np.asarray(per[lvl]["hi"], dtype=float) for lvl in levels]
        # Center / se are level-independent (symmetric CI); recover them from level 0.
        jhat = 0.5 * (lo[0] + hi[0])
        se = 0.5 * (hi[0] - lo[0]) / z[0]
        results.append({
            "N": int(N),
            "J_hat_N": jhat.tolist(),
            "se": se.tolist(),
            "lo": [col.tolist() for col in lo],
            "hi": [col.tolist() for col in hi],
        })

    data = {
        "algorithm": "coverage_intervals",
        "ci": (meta or {}).get("ci", "plugin"),
        "levels": levels,
        "n_ref": int(study["n_ref"]),
        "f_ref": float(study["f_ref"]),
        "R": int(study["R"]),
        "sample_sizes": [int(N) for N in study["sample_sizes"]],
        "meta": meta,
        "results": results,
    }
    with open(json_path, "w") as fh:
        json.dump(data, fh)   # compact (no indent): ~3.3 MB at R=5000, scales with R
    return json_path


def load_coverage_intervals(path):
    """Read back what ``save_coverage_intervals`` wrote, as numpy arrays.

    Returns the JSON dict with each ``results`` block's payload converted:
    ``lo``/``hi`` become ``(n_levels, R)`` arrays whose rows align with ``levels``,
    and ``J_hat_N``/``se`` become ``(R,)`` arrays.  The cover/no-cover indicator is
    not stored -- derive it from the endpoints as
    ``(lo <= data["f_ref"]) & (data["f_ref"] <= hi)``.
    """
    with open(path) as fh:
        data = json.load(fh)
    for res in data["results"]:
        res["J_hat_N"] = np.asarray(res["J_hat_N"], dtype=float)
        res["se"] = np.asarray(res["se"], dtype=float)
        res["lo"] = np.asarray(res["lo"], dtype=float)
        res["hi"] = np.asarray(res["hi"], dtype=float)
    return data
