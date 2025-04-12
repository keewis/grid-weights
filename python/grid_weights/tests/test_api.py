import pytest

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
