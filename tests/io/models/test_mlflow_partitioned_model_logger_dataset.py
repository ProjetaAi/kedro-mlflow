from pathlib import Path

import mlflow
import pandas as pd
import pytest
from kedro.io.core import DataSetError
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from sklearn.linear_model import LinearRegression

from kedro_mlflow.io.models import MlflowPartitionedModelLoggerDataSet


@pytest.fixture
def linreg_model_a():
    linreg_model = LinearRegression()
    linreg_model.fit(
        X=pd.DataFrame(data=[[1, 2], [3, 4]], columns=["a", "b"]),
        y=pd.Series(data=[5, 10]),
    )
    return linreg_model


@pytest.fixture
def linreg_model_b():
    linreg_model = LinearRegression()
    linreg_model.fit(
        X=pd.DataFrame(data=[[1, 2], [3, 4]], columns=["a", "b"]),
        y=pd.Series(data=[3, 7]),
    )
    return linreg_model


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


def sklearn_ds(
    kwargs: dict = None, data_set: dict = None
) -> MlflowPartitionedModelLoggerDataSet:
    return MlflowPartitionedModelLoggerDataSet.from_config(
        "test",
        {
            "type": "kedro_mlflow.io.models.MlflowPartitionedModelLoggerDataSet",
            "data_set": {
                "flavor": "mlflow.sklearn",
                **(data_set or {}),
            },
            **(kwargs or {}),
        },
    )


def test_partitioned_model_save_and_load(
    run_id: str,
    linreg_model_a: LinearRegression,
    linreg_model_b: LinearRegression,
):
    ds = sklearn_ds()
    ds.save({"a": linreg_model_a, "b": linreg_model_b})

    runs = mlflow.search_runs(filter_string=f"tags.{MLFLOW_PARENT_RUN_ID} = '{run_id}'")
    assert len(runs) == 2

    ds = sklearn_ds({"run_id": run_id})
    regressors_to_load = ds.load()

    assert regressors_to_load.keys() == {"a", "b"}

    regressors = {k: v() for k, v in regressors_to_load.items()}
    assert regressors["a"].predict([[1, 2]]) == pytest.approx([5])
    assert regressors["b"].predict([[1, 2]]) == pytest.approx([3])


def test_partitioned_model_save_multiple_times_and_load(
    run_id: str,
    linreg_model_a: LinearRegression,
    linreg_model_b: LinearRegression,
):
    ds = sklearn_ds()
    ds.save({"a": linreg_model_a, "b": linreg_model_b})

    ds2 = sklearn_ds(data_set={"artifact_path": "model2"})
    ds2.save({"a": linreg_model_b, "b": linreg_model_a})

    # assuring the datasets don't create new runs
    runs = mlflow.search_runs(filter_string=f"tags.{MLFLOW_PARENT_RUN_ID} = '{run_id}'")
    assert len(runs) == 2

    regs = sklearn_ds({"run_id": run_id}).load()
    regs2 = sklearn_ds({"run_id": run_id}, {"artifact_path": "model2"}).load()

    assert regs["a"]().predict([[1, 2]]) == regs2["b"]().predict([[1, 2]])
    assert regs2["b"]().predict([[1, 2]]) == regs["a"]().predict([[1, 2]])


def test_partitioned_model_dynamic_registered_name(
    run_id: str,
    linreg_model_a: LinearRegression,
    linreg_model_b: LinearRegression,
):
    ds = sklearn_ds(data_set={"save_args": {"registered_model_name": "test"}})
    ds.save({"a": linreg_model_a, "b": linreg_model_b})

    client = MlflowClient()
    models = client.list_registered_models()

    names = {f"{part}\\test" for part in ["a", "b"]}
    assert all(model.name in names for model in models)

    regressors = {
        model.name: mlflow.sklearn.load_model(f"models:/{model.name}/latest")
        for model in models
    }
    assert regressors["a\\test"].predict([[1, 2]]) == pytest.approx([5])
    assert regressors["b\\test"].predict([[1, 2]]) == pytest.approx([3])


def test_partitioned_model_validation():
    with pytest.raises(DataSetError):
        sklearn_ds(data_set={"flavor": "mlflow.whoops"})


def test_partitioned_model_create_run(
    linreg_model_a: LinearRegression, linreg_model_b: LinearRegression
):
    ds = sklearn_ds()
    ds.save({"a": linreg_model_a, "b": linreg_model_b})

    run_id = ds.parent.info.run_id
    runs = mlflow.search_runs(filter_string=f"tags.{MLFLOW_PARENT_RUN_ID} = '{run_id}'")
    assert len(runs) == 2

    ds = sklearn_ds({"run_id": run_id})
    regressors_to_load = ds.load()

    assert regressors_to_load.keys() == {"a", "b"}

    regressors = {k: v() for k, v in regressors_to_load.items()}
    assert regressors["a"].predict([[1, 2]]) == pytest.approx([5])
    assert regressors["b"].predict([[1, 2]]) == pytest.approx([3])


def test_partitioned_model_normalized_names(
    run_id: str, linreg_model_a: LinearRegression
):
    ds = sklearn_ds(data_set={"save_args": {"registered_model_name": "test"}})
    ds.save({"a/b/c": linreg_model_a})

    runs = mlflow.search_runs(filter_string=f"tags.{MLFLOW_PARENT_RUN_ID} = '{run_id}'")
    assert len(runs) == 1
    assert runs[f"tags.{MLFLOW_RUN_NAME}"][0] == "a\\b\\c"

    client = MlflowClient()
    models = client.list_registered_models()
    assert models[0].name == "a\\b\\c\\test"
