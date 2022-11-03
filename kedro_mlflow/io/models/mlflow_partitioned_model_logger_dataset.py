from typing import Any, Dict

from kedro_mlflow.io.partitioned import MlflowPartitionedDataSet


class MlflowPartitionedModelLoggerDataSet(MlflowPartitionedDataSet):
    """Wrapper for ``MlflowModelLoggerDataSet`` to work dynamically."""

    def __init__(
        self,
        data_set: Dict[str, Any] = {},
        credentials: Dict[str, Any] = None,
        load_args: Dict[str, Any] = None,
        run_id: str = None,
    ):
        """Initialize the Kedro ``MlflowPartitionedModelLoggerDataSet``.

        Args:
            data_set (Dict[str, Any], optional): Underlying MlflowModelLoggerDataSet
                configuration. Defaults to {}.
            credentials (Dict[str, Any], optional): Credentials to access the underlying
                filesystem. Defaults to None.
            load_args (Dict[str, Any], optional): Additional arguments to pass to the
                underlying dataset. Defaults to None.
            pyfunc_workflow (str, optional): Either `python_model` or `loader_module`.
                See https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#workflows.
            run_id (str, optional): The run ID of the parent run, if not
                specified equals to top of the active runs stack. Defaults to None.
        """
        super().__init__(
            {
                "type": "kedro_mlflow.io.models.MlflowModelLoggerDataSet",
                **data_set,
            },
            credentials,
            load_args,
            run_id,
            {"save_args.registered_model_name"},
        )
        self._validate()

    def _validate(self):
        self._new_child_dataset("check")
