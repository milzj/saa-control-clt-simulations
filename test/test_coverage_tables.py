"""Solve-free checks for the two revision coverage tables."""

import importlib.util
import json
from pathlib import Path

import pytest
from scipy.stats import binom


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coverage_tables", ROOT / "scripts/fed_batch_reactor/coverage_tables.py")
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)

EXPECTED = {
    "z": [(8392, 8949, 9533), (8721, 9220, 9708), (8826, 9345, 9840),
          (8882, 9404, 9850), (8837, 9408, 9859)],
    "t": [(9037, 9502, 9893), (9005, 9479, 9884), (8977, 9500, 9900),
          (8956, 9469, 9885), (8868, 9441, 9878)],
}


@pytest.mark.parametrize("method", ["z", "t"])
def test_archived_counts_and_generated_tables(method):
    data = json.loads((ROOT / "results/revision/coverage_replications.json").read_text())
    assert data["R"] == 10000
    assert data["n_ref"] == 4096
    assert data["f_ref"] == -32.09682940481349
    rows = tables.coverage_rows(data, method)
    assert [row["N"] for row in rows] == [8, 16, 32, 64, 128]
    for row, counts in zip(rows, EXPECTED[method]):
        assert tuple(v["count"] for v in row["values"]) == counts
        for value in row["values"]:
            # Independent characterization of the binomial lower-bound formula.
            assert binom.sf(value["count"] - 1, data["R"], value["lower"]) == pytest.approx(1e-6)
    filename = "coverage_plugin.tex" if method == "z" else "coverage_plugin_t.tex"
    assert tables.latex_table(data, method) == (ROOT / "results/revision" / filename).read_text()


def test_inclusive_endpoints_and_extreme_counts():
    data = {"R": 2, "f_ref": 0.0, "results": [
        {"N": 8, "J_hat_N": [0.0, 0.0], "se": [0.0, 0.0]}]}
    values = tables.coverage_rows(data, "z")[0]["values"]
    assert all(v["count"] == 2 and v["lower"] == pytest.approx(1e-3) for v in values)
    data["f_ref"] = 1.0
    assert all(v["count"] == 0 and v["lower"] == 0.0
               for v in tables.coverage_rows(data, "t")[0]["values"])


@pytest.mark.parametrize("field,value", [
    ("N", 1), ("se", [-1.0, 1.0]), ("se", [1.0]),
    ("J_hat_N", [float("nan"), 0.0]),
])
def test_invalid_replications_rejected(field, value):
    block = {"N": 8, "J_hat_N": [0.0, 0.0], "se": [1.0, 1.0]}
    block[field] = value
    with pytest.raises(ValueError):
        tables.coverage_rows({"R": 2, "f_ref": 0.0, "results": [block]}, "t")
