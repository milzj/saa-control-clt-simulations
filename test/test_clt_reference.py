"""The CLT reference-solve companion file.

``ensemblecontrol.save_clt_run`` keeps only the reference VALUE (f_ref) and drops the
reference CONTROL (w_ref), so ``clt.json`` cannot say what the reference solution was.
``save_clt_reference`` persists it alongside.  Pure I/O -- no solver.
"""

import json

import numpy as np

from saa_clt.outputs import save_clt_reference

Q = 50


def _study():
    """The shape ensemblecontrol.clt_replication_study returns."""
    return {"values_by_N": {8: np.zeros(3)},
            "f_ref": -32.042858846701826,
            "w_ref": np.linspace(0.0, 1.0, Q),
            "n_ref": 4096,
            "q": Q}


def test_saves_reference_control_value_and_sigma(tmp_path):
    out = tmp_path / "clt_reference.json"
    study = _study()
    save_clt_reference(study, str(out), sigma_ref=3.8254,
                       meta={"model": "FedBatchReactor", "seed": 1234, "R": 300})

    data = json.loads(out.read_text())
    assert data["algorithm"] == "clt_reference"
    assert data["n_ref"] == 4096
    assert data["f_ref"] == study["f_ref"]
    assert data["q"] == Q
    assert data["sigma_ref"] == 3.8254
    # the reference CONTROL itself -- the thing save_clt_run drops
    assert len(data["w_ref"]) == Q
    np.testing.assert_allclose(data["w_ref"], study["w_ref"])
    assert data["meta"]["seed"] == 1234   # + R: enough to re-draw the reference sample


def test_sigma_ref_optional(tmp_path):
    out = tmp_path / "clt_reference.json"
    save_clt_reference(_study(), str(out))
    assert json.loads(out.read_text())["sigma_ref"] is None


def test_w_ref_is_flattened(tmp_path):
    # clt_replication_study returns w_ref as ndarray; a column shape must not
    # produce a nested list.
    out = tmp_path / "clt_reference.json"
    study = _study()
    study["w_ref"] = study["w_ref"].reshape(Q, 1)
    save_clt_reference(study, str(out))
    w = json.loads(out.read_text())["w_ref"]
    assert len(w) == Q and all(isinstance(v, float) for v in w)
