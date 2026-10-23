"""Reporting must retain signed history weights and include optional models."""

import pandas as pd

from rl_ladder.collect import _collect_param_recovery
from rl_ladder.figures import model_order
from rl_ladder.service import context


def test_recovery_summary_preserves_negative_choice_weights(tmp_path):
    unit = {"variant": "test", "model": "C", "rep": 0, "template": "synthetic"}
    result = {"true": {"beta_c": -1.25}, "fit": {"beta_c": -0.75},
              "accuracy": 0.5, "optimizer_miss": False, "at_bound": []}
    _collect_param_recovery([(unit, {"result": result})], tmp_path, "", context(), False)
    row = pd.read_csv(tmp_path / "param_recovery.csv").iloc[0]
    assert row.true_z == -1.25
    assert row.fit_z == -0.75


def test_figure_order_keeps_feedback_and_optional_models():
    models = ["B+F+C", "S+F+Cf", "S+R+F+C", "F+C"]
    assert set(model_order(models)) == set(models)
