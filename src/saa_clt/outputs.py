"""Output-path helper: route every driver's results to a single top-level
``results/<example>/<study>/<stamp>/`` tree at the repository root.

Each driver lives in ``scripts/<example>/`` and calls
``study_dir(__file__, "<study>", stamp)``; the example name is taken from the
driver's parent folder and the repository root is located by walking up to the
directory that contains ``pyproject.toml``.
"""

import os

__all__ = ["repo_root", "example_name", "study_dir"]


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
