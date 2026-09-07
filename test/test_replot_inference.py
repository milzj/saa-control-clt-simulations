import importlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replot_inference.py"
SPEC = importlib.util.spec_from_file_location("replot_inference", SCRIPT)
replot_inference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replot_inference)


def test_exact_subsampling_plotter_restores_dependency_helper():
    plotting = importlib.import_module("ensemblecontrol.inference_plotting")
    original = plotting.subsampling_ci_from_deltas

    with replot_inference._exact_subsampling_plotter():
        assert (
            plotting.subsampling_ci_from_deltas
            is replot_inference.subsampling_cis_from_deltas
        )

    assert plotting.subsampling_ci_from_deltas is original

    with pytest.raises(RuntimeError, match="test restoration"):
        with replot_inference._exact_subsampling_plotter():
            raise RuntimeError("test restoration")

    assert plotting.subsampling_ci_from_deltas is original


def test_subsampling_only_skips_plugin_figures(tmp_path, monkeypatch):
    run_dir = tmp_path / "2026-01-02T03-04-05"
    run_dir.mkdir()
    (run_dir / replot_inference.PLUGIN_JSON).write_text("{}")
    (run_dir / replot_inference.SUB_JSON).write_text("{}")

    plugin_run = {"kind": "plugin"}
    sub_run = {"kind": "subsampling"}
    monkeypatch.setattr(
        replot_inference.ensemblecontrol, "load_plugin_run", lambda _path: plugin_run
    )
    monkeypatch.setattr(
        replot_inference.ensemblecontrol, "load_subsampling_run", lambda _path: sub_run
    )
    monkeypatch.setattr(replot_inference, "_plugin_cis", lambda _run: [])
    monkeypatch.setattr(replot_inference, "_subsampling_cis", lambda _run: [])
    monkeypatch.setattr(
        replot_inference.ensemblecontrol, "value_ylim_across", lambda _cis: None
    )

    def fail_plugin(*_args, **_kwargs):
        pytest.fail("plug-in figures must not be rendered")

    calls = []

    def fake_subsampling(run, **kwargs):
        calls.append((run, kwargs))
        return ["subsampling.pdf", "subsampling.png"]

    monkeypatch.setattr(replot_inference.ensemblecontrol, "plot_plugin", fail_plugin)
    monkeypatch.setattr(replot_inference, "_plot_subsampling", fake_subsampling)
    written = replot_inference.replot(str(run_dir), subsampling_only=True)

    assert written == ["subsampling.pdf", "subsampling.png"]
    assert len(calls) == 1
    assert calls[0][0] is sub_run
    assert calls[0][1]["outdir"] == str(run_dir)
    assert calls[0][1]["stamp"] == run_dir.name
