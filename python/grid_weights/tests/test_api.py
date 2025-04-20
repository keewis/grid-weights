import pytest
import xarray as xr

from grid_weights import api


class TestAlgorithms:
    @pytest.mark.parametrize(
        ["variables", "error"],
        (
            ({"a": "conservative", "b": "conservative"}, None),
            ({"a": "not-existing"}, ValueError("unknown algorithms: not-existing")),
            (
                {"a": "not-existing", "b": "conservative"},
                ValueError("unknown algorithms: not-existing"),
            ),
            (
                {"a": "not-existing", "b": "not-existing"},
                ValueError("unknown algorithms: not-existing"),
            ),
            (
                {"a": "not-existing", "b": "invalid"},
                ValueError("unknown algorithms: not-existing, invalid"),
            ),
        ),
    )
    def test_init_errors(self, variables, error, monkeypatch):
        if error is None:
            config = api.Algorithms(variables)
            assert config.variables == variables
        else:
            with pytest.raises(type(error), match=error.args[0]):
                api.Algorithms(variables)

    @pytest.mark.parametrize(
        ["ds", "variables", "default", "variable_kwargs", "expected"],
        (
            (
                xr.Dataset({"a": ("x", [1])}),
                {"a": "conservative"},
                None,
                {},
                {"a": "conservative"},
            ),
            (
                xr.Dataset({"a": ("x", [1])}),
                None,
                None,
                {"a": "conservative"},
                {"a": "conservative"},
            ),
            (
                xr.Dataset({"a": ("x", [1])}),
                None,
                "conservative",
                {},
                {"a": "conservative"},
            ),
            (
                xr.Dataset({"a": ("x", [1]), "b": ("x", [2])}),
                None,
                "conservative",
                {},
                {"a": "conservative", "b": "conservative"},
            ),
        ),
    )
    def test_by_variable(self, ds, variables, default, variable_kwargs, expected):
        actual = api.Algorithms.by_variable(ds, variables, default, **variable_kwargs)

        assert actual.variables == expected
