from typing import Dict

from azureml.core import Workspace

from kedro_mlflow.config import KedroMlflowConnection


class AzureMLConnection(KedroMlflowConnection):
    """Parses `azureml` as a keyword to get the tracking URI from AzureML workspace."""

    def tracking_uri(
        self,
        credentials: Dict[str, str],
        options: Dict[str, str],
    ) -> str:
        """URI to use for tracking."""
        ws = Workspace(
            subscription_id=self.getkey(
                options, "subscription_id", "AZUREML_SUBSCRIPTION_ID"
            ),
            resource_group=self.getkey(
                options, "resource_group", "AZUREML_RESOURCE_GROUP"
            ),
            workspace_name=self.getkey(
                options, "workspace_name", "AZUREML_WORKSPACE_NAME"
            ),
        )
        return ws.get_mlflow_tracking_uri()


azureml_connection = AzureMLConnection()
