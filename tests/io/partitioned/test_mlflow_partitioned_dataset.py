from pathlib import Path
from typing import Literal

import mlflow
import pytest

from kedro_mlflow.io.partitioned import MlflowPartitionedDataSet


@pytest.fixture
def tracking_uri(tmp_path: Path):
    # Requires sqlite in order to use model registry
    folder = tmp_path / "mlruns"
    folder.mkdir(parents=True)
    tracking_uri = f"sqlite:///{(folder / 'sqlite.db').as_posix()}"
    return tracking_uri


@pytest.fixture
def run_id(tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    run_id = mlflow.start_run(nested=True).info.run_id
    return run_id


def dataset(
    type: Literal["Metric", "MetricHistory", "Metrics"],
    kwargs: dict = None,
    data_set: dict = None,
):
    return MlflowPartitionedDataSet.from_config(
        "test",
        {
            "type": "kedro_mlflow.io.partitioned.MlflowPartitionedDataSet",
            "data_set": {
                "type": f"kedro_mlflow.io.metrics.Mlflow{type}DataSet",
                **(data_set or {}),
            },
            **(kwargs or {}),
        },
    )


def test_partitioned_metric_save_and_load(run_id: str):
    ds = dataset("Metric", data_set={"key": "mse"})
    ds.save({"a": 1, "b": 2})

    ds = dataset("Metric", data_set={"key": "mse"}, kwargs={"run_id": run_id})
    metrics = ds.load()
    assert metrics.keys() == {"a", "b"}
    assert metrics["a"]() == 1
    assert metrics["b"]() == 2


def test_partitioned_metric_history_save_and_load(run_id: str):
    ds = dataset("MetricHistory", data_set={"key": "mae"})
    ds.save({"a": [1, 2, 3], "b": [2, 3, 4]})

    ds = dataset("MetricHistory", data_set={"key": "mae"}, kwargs={"run_id": run_id})
    metrics = ds.load()
    assert metrics.keys() == {"a", "b"}
    assert metrics["a"]() == [1, 2, 3]
    assert metrics["b"]() == [2, 3, 4]


def test_partitioned_metrics_save_and_load(run_id: str):
    data = {
        "a": {
            "mse": [{"value": 1, "step": 1}],
            "mae": [{"value": 1, "step": 1}, {"value": 2, "step": 2}],
        },
        "b": {
            "mse": [{"value": 2, "step": 1}],
            "rmse": [{"value": 2, "step": 1}, {"value": 3, "step": 2}],
        },
    }

    ds = dataset("Metrics")
    ds.save(data)

    ds = dataset("Metrics", kwargs={"run_id": run_id})
    metrics_to_load = ds.load()

    assert metrics_to_load.keys() == {"a", "b"}

    metrics = {k: v() for k, v in metrics_to_load.items()}
    assert metrics["a"].keys() == {"mse", "mae"}
    assert metrics["b"].keys() == {"mse", "rmse"}
    for partition in data:
        for metric in data[partition]:
            original = data[partition][metric]
            loaded = metrics[partition][metric]
            loaded = [loaded] if isinstance(loaded, dict) else loaded
            for ldata, odata in zip(original, loaded):
                assert ldata["value"] == odata["value"]
