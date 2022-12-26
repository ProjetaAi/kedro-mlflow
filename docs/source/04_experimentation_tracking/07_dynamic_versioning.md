# Dynamic Versioning

Sometimes you have a variable number of models or metrics to save in your pipeline. For example, let's say you're training a model per store in your company. In this case you may declare a dataset for each store model, but this would require manual intervention and it might be inviable if you have a lot of stores. Because of that, ``kedro-mlflow`` provides a way to dynamically use the its datasets, similarly to Kedro's partitioned datasets.

## How to use dynamic versioning?

If you've ever used Kedro's partitioned dataset, you know for saving a dataset you need to pass a dictionary with the partition (subpath) as key and the data as value. However, because we are not dealing with folders, but with ``mlflow``, the keys are used to name the child runs nested in the current run. For example, if you have a dataset with the following configuration:

```yaml
my_dataset_to_version:
    type: kedro_mlflow.io.partitioned.MlflowPartitionedDataSet
    data_set:
        type: kedro_mlflow.io.metrics.MlflowMetricDataSet
        key: my_metric
```

and you save it with the following code:

```python
my_dataset_to_version.save({"store_1": 0.5, "store_2": 0.7})
```

you will have two child runs nested in the current run, one for each store. The first one will have the name ``store_1`` and the second one will have the name ``store_2``. The metric ``my_metric`` will be logged in each child run with the corresponding value.

## Model registry

Besides the ``MlflowPartitionedDataSet``, kedro-mlflow provides a ``MlflowPartitionedModelLoggerDataSet`` which is a sugar for the ``MlflowModelLoggerDataSet`` combined with ``MlflowPartitionedDataSet`` that not only logs the artifacts in child runs, but also uses the partition name as the model name in mlflow model registry. This means that if you have a dataset with the following configuration:

```yaml
my_model_to_version:
    type: kedro_mlflow.io.models.MlflowPartitionedModelLoggerDataSet
    save_args:
        registered_model_name: "test"
```

and you save it with the following code:

```python
my_model_to_version.save({"store_1": model_1, "store_2": model_2})
```

you will have two child runs nested in the current run, one for each store. The first one will have the name ``store_1`` and the second one will have the name ``store_2``. While the registered models will have the name ``store_1\test`` and ``store_2\test`` respectively.

```{note}
Every partition key normalizes `/` to `\` in order to not break mlflow URIs.
```
