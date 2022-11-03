from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Iterable, Set, Type, Union

import mlflow
import mlflow.tracking.fluent as mlflow_fluent
import pandas as pd
from kedro.io import AbstractDataSet, DataSetError, PartitionedDataSet
from mlflow.entities.run import Run
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

from kedro_mlflow.utils import _flatten_dict, _unflatten_dict

PREF_MLFLOW_PARENT_RUN_ID = f"tags.{MLFLOW_PARENT_RUN_ID}"
PREF_MLFLOW_RUN_NAME = f"tags.{MLFLOW_RUN_NAME}"
MLFLOW_RUN_ID = "run_id"


class _ProtectedDict(dict):
    """A dict that doesn't allow overwriting keys if protected."""

    def __init__(self, data: dict, protected: Iterable[str]):
        super().__init__(data)
        self.protected = set(protected)

    def __setitem__(self, key, value):
        if key not in self.protected:
            super().__setitem__(key, value)  # pragma: no cover

    def __deepcopy__(self, _) -> "_ProtectedDict":
        return _ProtectedDict({**self}, self.protected)


class MlflowPartitionedDataSet(PartitionedDataSet):
    """Wrapper for Kedro Mlflow datasets to allow dynamic datasets."""

    def __init__(
        self,
        data_set: Union[str, Type[AbstractDataSet], Dict[str, Any]],
        credentials: Dict[str, Any] = None,
        load_args: Dict[str, Any] = None,
        run_id: str = None,
        dynamic_parameters: Set[str] = None,
    ):
        """Initialize the Kedro MlflowPartitionedDataSet.

        Args:
            data_set (Union[str, Type[AbstractDataSet], Dict[str, Any]]): The dataset for each partition.
            credentials (Dict[str, Any], optional): Credentials to access the underlying filesystem.
                Defaults to None.
            load_args (Dict[str, Any], optional): Additional arguments to pass to the underlying dataset.
                Defaults to None.
            parent_run_id (str, optional): The run ID of the parent run, if not specified
                equals to top of the active runs stack. Defaults to None.
            dynamic_parameters (Set[str], optional): Parameters that are dynamic and should be
                replaced with the partition name. Defaults to None.
        """
        super().__init__("", data_set, "", "", credentials, load_args, None, False)
        self._run_id = run_id
        self._dynamic_parameters = dynamic_parameters or set()

    @property
    def parent(self) -> Run:
        """Gets the parent to run the child runs under."""
        mlflow.active_run()
        if self._run_id:
            return mlflow.get_run(self._run_id)
        else:
            if mlflow_fluent._active_run_stack:
                # can't take the top of the stack because of race conditions
                return mlflow_fluent._active_run_stack[0]
            else:
                return mlflow.start_run(nested=True)

    def _subname(self, partition: str, suffix: str, sep: str = "\\") -> str:
        """Gets a subname from a partition and a property."""
        return sep.join((PurePosixPath(partition) / suffix).parts)

    def _normalize(self, name: str, sep: str = "\\") -> str:
        return self._subname(name, "", sep=sep)

    def _new_child_dataset(
        self, child_name: str, extra: Dict[str, Any] = None
    ) -> AbstractDataSet:
        """Creates a new child dataset from configuration."""
        extra = extra or {}
        config = _flatten_dict({**self._dataset_config, **extra})

        for param in self._dynamic_parameters:
            if param in config:
                config[param] = self._subname(child_name, config[param])

        config = _unflatten_dict(config)
        return self._dataset_type(**config)

    def find_children(self) -> Dict[str, str]:
        """Finds all child runs of the parent run.

        Returns:
            Dict[str, str]: A dictionary of run IDs to run names.
        """
        runs: pd.DataFrame = mlflow.search_runs(
            filter_string=f'{PREF_MLFLOW_PARENT_RUN_ID}="{self.parent.info.run_id}"'
        )
        if runs.empty:
            raise DataSetError(
                f"No child runs found for parent run '{self.parent.info.run_id}'"
            )

        return (
            runs[[MLFLOW_RUN_ID, PREF_MLFLOW_RUN_NAME]]
            .set_index(PREF_MLFLOW_RUN_NAME)[MLFLOW_RUN_ID]
            .to_dict()
        )

    def start_child_run(self, partition: str) -> Run:
        """Creates a new child run."""
        try:
            children = self.find_children()
            if partition in children:
                return mlflow.start_run(run_id=children[partition], nested=True)
        except DataSetError:
            pass

        tags = self.parent.data.tags
        tags.pop(MLFLOW_RUN_NAME, None)
        tags[MLFLOW_PARENT_RUN_ID] = self.parent.info.run_id

        return mlflow.start_run(
            run_name=partition,
            nested=True,
            tags=_ProtectedDict(tags, {MLFLOW_PARENT_RUN_ID}),
        )

    def _save(self, data: Dict[str, Any]) -> None:
        for partition, child_data in data.items():
            child_name = self._normalize(partition)
            with self.start_child_run(child_name):
                self._new_child_dataset(child_name).save(child_data)

    def _load(self) -> Dict[str, Callable[..., Any]]:
        return {
            child_name: self._new_child_dataset(child_name, {"run_id": child_id}).load
            for child_name, child_id in self.find_children().items()
        }
