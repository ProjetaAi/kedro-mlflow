from collections import Counter
from itertools import chain
from typing import Any, Dict, Set, cast

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node

from kedro_mlflow.io.partitioned import MlflowPartitionedDataSet


class MlflowPartitionedHook:
    """Fixes ``asyncio`` problems when using ``MlflowPartitionedDataSet``."""

    def __init__(self):
        self._datasets: Set[str] = set()
        self._save_history: Set[str] = set()

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        feed_dict: Dict[str, Any],
        save_version: str,
        load_versions: Dict[str, str],
    ) -> None:
        for name, dataset in catalog._data_sets.items():
            if isinstance(dataset, MlflowPartitionedDataSet):
                self._datasets.add(name)

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None:
        if is_async:
            # finds partitions that may generate race condition
            datasets = self._datasets & set(outputs.keys())
            partitions = Counter(chain(*(outputs[name].keys() for name in datasets)))
            problematic = {part for part in partitions if partitions[part] > 1}

            # creates child runs for partitions with risk of race condition
            for dataset_name in datasets:
                dataset = cast(
                    MlflowPartitionedDataSet, catalog._data_sets[dataset_name]
                )

                problematic -= self._save_history
                runs_to_start = problematic & set(outputs[dataset_name].keys())
                for partition in runs_to_start:
                    with dataset.start_child_run(partition):
                        self._save_history.add(partition)


mlflow_partitioned_hook = MlflowPartitionedHook()
