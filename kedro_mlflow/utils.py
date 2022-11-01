from pathlib import Path
from typing import Dict, List, Union

import flatten_dict


def _parse_requirements(path: Union[str, Path], encoding="utf-8") -> List:
    with open(path, mode="r", encoding=encoding) as file_handler:
        requirements = [
            x.strip() for x in file_handler if x.strip() and not x.startswith("-r")
        ]
    return requirements


def _flatten_dict(d: Dict, recursive: bool = True, sep: str = ".") -> Dict:
    def reducer(k1: str, k2: str):
        return f"{k1}{sep}{k2}" if k1 else str(k2)

    return flatten_dict.flatten(
        d, reducer=reducer, max_flatten_depth=(None if recursive else 2)
    )


def _unflatten_dict(d: Dict, sep: str = ".") -> Dict:
    def splitter(k: str):
        return k.split(sep)

    return flatten_dict.unflatten(d, splitter=splitter)
