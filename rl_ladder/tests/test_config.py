import pytest

from rl_ladder.config import apply_overrides, load_default, parse_cli_overrides
from rl_ladder.models.registry import make_spec, parents
from rl_ladder.service import context


def test_cli_overrides_are_typed_and_checked():
    o = parse_cli_overrides(["--fit.n_starts=40", "--preprocess.correction", "include",
                             "--model.session_reset=[wsls, choice]"])
    assert o == {"fit.n_starts": 40, "preprocess.correction": "include",
                 "model.session_reset": ["wsls", "choice"]}
    cfg = apply_overrides(load_default(), o)
    assert cfg["fit"]["n_starts"] == 40
    with pytest.raises(KeyError):
        apply_overrides(load_default(), {"fit.nstarts": 40})


def test_registry_rules():
    with pytest.raises(ValueError):
        make_spec("bad", {"stimulus": "S", "action": "gamma"})
    m = context().models
    assert m["M2"].free == ("eps",)
    assert parents(m["S+R+C"], m) == ["S+C", "S+R"]
    assert parents(m["A+R"], m) == ["A0"]
    assert [m[x].k for x in context().sets["core"]] == [1, 1, 4, 6, 6, 8, 3, 4, 5, 6]
