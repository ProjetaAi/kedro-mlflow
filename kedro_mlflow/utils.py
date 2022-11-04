import importlib.metadata
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Union


def _parse_requirements(path: Union[str, Path], encoding="utf-8") -> List:
    with open(path, mode="r", encoding=encoding) as file_handler:
        requirements = [
            x.strip() for x in file_handler if x.strip() and not x.startswith("-r")
        ]
    return requirements


@lru_cache(maxsize=None)
def _load_plugins(entry_point: str) -> Dict[str, Callable[[], Any]]:
    return {
        plugin.name: plugin.load
        for plugin in importlib.metadata.entry_points()[entry_point]
    }
