import mlflow
import pytest
from kedro.io import DataCatalog
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

from kedro_mlflow.framework.hooks import MlflowPartitionedHook


@pytest.fixture
def catalog():
    return DataCatalog.from_config(
        {
            "metric": {
                "type": "kedro_mlflow.io.partitioned.MlflowPartitionedDataSet",
                "data_set": {
                    "type": "kedro_mlflow.io.metrics.MlflowMetricsDataSet",
                    "key": "foo",
                },
            },
            "metric_history": {
                "type": "kedro_mlflow.io.partitioned.MlflowPartitionedDataSet",
                "data_set": {
                    "type": "kedro_mlflow.io.metrics.MlflowMetricsDataSet",
                    "key": "bar",
                },
            },
            "csv": {
                "type": "pandas.CSVDataSet",
                "filepath": "data/01_raw/iris.csv",
            },
        }
    )


@pytest.fixture
def partitioned_hook():
    return MlflowPartitionedHook()


def test_partitioned_hook_find_unsafe_datasets(
    partitioned_hook: MlflowPartitionedHook, catalog: DataCatalog
):
    partitioned_hook.after_catalog_created(catalog, None, None, None, None, None)
    assert partitioned_hook._datasets == {"metric", "metric_history"}


def test_partitioned_hook_create_runs(
    partitioned_hook: MlflowPartitionedHook, catalog: DataCatalog
):
    run = mlflow.start_run()
    partitioned_hook.after_catalog_created(catalog, None, None, None, None, None)
    partitioned_hook.after_node_run(
        None,
        catalog,
        None,
        {
            "metric": {"a": 1, "b": 2},
            "metric_history": {"a": [1, 2], "b": [2, 3]},
        },
        True,
        None,
    )

    runs = mlflow.search_runs(
        filter_string=f"tags.{MLFLOW_PARENT_RUN_ID}='{run.info.run_id}'"
    )
    assert len(runs) == 2
    assert set(runs[f"tags.{MLFLOW_RUN_NAME}"]) == {"a", "b"}
