import copy
from typing import Any, Dict, Optional, Union

import mlflow
from kedro.io.core import DataSetError
from mlflow.models import ModelSignature
from mlflow.models.utils import ModelInputExample

from kedro_mlflow.io.models.mlflow_abstract_model_dataset import (
    MlflowAbstractModelDataSet,
)


class MlflowModel:
    """Dynamic ``Mlflow`` model args for the ``log_model`` function."""

    def __init__(
        self,
        model: Any,
        signature: Optional[ModelSignature] = None,
        input_example: Optional[ModelInputExample] = None,
        await_registration_for: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the MlflowModel."""
        self.model = model
        self.kwargs = copy.deepcopy(kwargs)

        if signature is not None:
            self.kwargs["signature"] = signature
        if input_example is not None:
            self.kwargs["input_example"] = input_example
        if await_registration_for is not None:
            self.kwargs["await_registration_for"] = await_registration_for


class MlflowModelLoggerDataSet(MlflowAbstractModelDataSet):
    """Wrapper for saving, logging and loading for all MLflow model flavor."""

    def __init__(
        self,
        flavor: str,
        run_id: Optional[str] = None,
        artifact_path: Optional[str] = None,
        pyfunc_workflow: Optional[str] = None,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Kedro MlflowModelDataSet.

        Parameters are passed from the Data Catalog.

        During save, the model is first logged to MLflow.
        During load, the model is pulled from MLflow run with `run_id`.

        Args:
            flavor (str): Built-in or custom MLflow model flavor module.
                Must be Python-importable.
            run_id (Optional[str], optional): MLflow run ID to use to load
                the model from or save the model to. Defaults to None.
            artifact_path (str, optional): the run relative path to
                the model.
            pyfunc_workflow (str, optional): Either `python_model` or `loader_module`.
                See https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#workflows.
            load_args (Dict[str, Any], optional): Arguments to `load_model`
                function from specified `flavor`. Defaults to None.
            save_args (Dict[str, Any], optional): Arguments to `log_model`
                function from specified `flavor`. Defaults to None.

        Raises:
            DataSetError: When passed `flavor` does not exist.
        """
        super().__init__(
            filepath="",
            flavor=flavor,
            pyfunc_workflow=pyfunc_workflow,
            load_args=load_args,
            save_args=save_args,
            version=None,
        )

        self._run_id = run_id
        self._artifact_path = artifact_path or "model"

        # drop the key which MUST be common to save and load and
        #  thus is instantiated outside save_args
        self._save_args.pop("artifact_path", None)

    @property
    def _safe_pyfunc_workflow(self) -> str:
        """Get the pyfunc workflow."""
        assert isinstance(
            self._pyfunc_workflow, str
        ), "'pyfunc_workflow' must be specified."
        return self._pyfunc_workflow

    @property
    def model_uri(self):
        run_id = None
        if self._run_id:
            run_id = self._run_id
        elif mlflow.active_run() is not None:
            run_id = mlflow.active_run().info.run_id
        if run_id is None:
            raise DataSetError(
                (
                    "To access the model_uri, you must either: "
                    "\n -  specifiy 'run_id' "
                    "\n - have an active run to retrieve data from"
                )
            )

        model_uri = f"runs:/{run_id}/{self._artifact_path}"

        return model_uri

    def _load(self) -> Any:
        """Loads an MLflow model from local path or from MLflow run.

        Returns:
            Any: Deserialized model.
        """

        # If `run_id` is specified, pull the model from MLflow.
        # TODO: enable loading from another mlflow conf (with a client with another tracking uri)
        # Alternatively, use local path to load the model.
        return self._mlflow_model_module.load_model(
            model_uri=self.model_uri, **self._load_args
        )

    def _save(self, model: Union[Any, MlflowModel]) -> None:
        """Save a model to local path and then logs it to MLflow.

        Args:
            model (Any): A model object supported by the given MLflow flavor.
        """
        if self._run_id:
            if mlflow.active_run():
                # it is not possible to log in a run which is not the current opened one
                raise DataSetError(
                    (
                        "'run_id' cannot be specified"
                        " if there is an mlflow active run."
                        "Run_id mismatch: "
                        f"\n - 'run_id'={self._run_id}"
                        f"\n - active_run id={mlflow.active_run().info.run_id}"
                    )
                )
            else:
                # if the run id is specified and there is no opened run,
                # open the right run before logging
                with mlflow.start_run(run_id=self._run_id):
                    self._save_model_in_run(model)
        else:
            # if there is no run_id, log in active run
            # OR open automatically a new run to log
            self._save_model_in_run(model)

    def _save_model_in_run(self, model: Union[Any, MlflowModel]):
        kwargs = {
            "artifact_path": self._artifact_path,
            **self._save_args,
        }
        if isinstance(model, MlflowModel):
            kwargs.update(model.kwargs)
            model = model.model

        if self._flavor == "mlflow.pyfunc":
            # PyFunc models utilise either `python_model` or `loader_module`
            # workflow. We we assign the passed `model` object to one of those keys
            # depending on the chosen `pyfunc_workflow`.
            kwargs[self._safe_pyfunc_workflow] = model
            if self._logging_activated:
                self._mlflow_model_module.log_model(**kwargs)
        else:
            # Otherwise we save using the common workflow where first argument is the
            # model object and second is the path.
            if self._logging_activated:
                self._mlflow_model_module.log_model(model, **kwargs)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            flavor=self._flavor,
            run_id=self._run_id,
            artifact_path=self._artifact_path,
            pyfunc_workflow=self._pyfunc_workflow,
            load_args=self._load_args,
            save_args=self._save_args,
        )
