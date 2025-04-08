import copy
from collections.abc import Hashable
from dataclasses import dataclass

import xarray as xr
from tlz.itertoolz import concat, groupby
from xarray.namedarray.utils import either_dict_or_kwargs


@dataclass
class Algorithms:
    algorithms: dict[str, list[Hashable]]

    @classmethod
    def by_variable(
        cls,
        ds: xr.Dataset,
        variables: dict[Hashable, str] = None,
        default: str = None,
        **variable_kwargs,
    ):
        passed_variables = either_dict_or_kwargs(
            variables, variable_kwargs, func_name="Algorithms.by_variable"
        )

        missing_variables = [
            name for name in ds.variables if name not in passed_variables
        ]
        if missing_variables and default is not None:
            raise ValueError(
                f"no configuration for {missing_variables} and no default set"
            )

        variables = passed_variables | {k: default for k in missing_variables}

        algorithms = groupby(lambda it: it[1], variables.items())
        print(algorithms)
        return cls(variables)

    @classmethod
    def by_algorithm(
        cls,
        ds: xr.Dataset,
        algorithms: dict[str, Hashable | list[Hashable]],
        default: str = None,
        **algorithm_kwargs,
    ):
        passed_algorithms = copy.deepcopy(
            either_dict_or_kwargs(
                algorithms, algorithm_kwargs, func_name="Algorithms.by_algorithm"
            )
        )

        all_variables = set(
            concat(
                [v] if isinstance(v, Hashable) else v
                for v in passed_algorithms.values()
            )
        )
        missing_variables = [name for name in ds.variables if name not in all_variables]
        if missing_variables and default is None:
            raise ValueError(
                f"no configuration for {missing_variables} and no default set"
            )

        default_algorithm = passed_algorithms.setdefault(default, [])
        default_algorithm.extend(missing_variables)

        return cls(passed_algorithms)
