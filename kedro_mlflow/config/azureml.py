from typing import Dict

from azureml.core import Workspace

from kedro_mlflow.config import KedroMlflowConnection


class AzureMLConnection(KedroMlflowConnection):
    """Parses `azureml` as a keyword to get the tracking URI from AzureML workspace.

    Credentials:
        No credentials are required. But the user must be logged in using the Azure CLI.

    Connection Options:
        This connection requires the following options to be set on the
        `server.connection` field in `mlflow.yml`, or in the environment
        variables:
        - `subscription_id` or `AZUREML_SUBSCRIPTION_ID`: Azure subscription ID
        - `resource_group` or `AZUREML_RESOURCE_GROUP`: Azure resource group
        - `workspace_name` or `AZUREML_WORKSPACE_NAME`: Azure workspace name

    Output:
        With these settings defined, the tracking URI and the registry URI will
        automatically be extracted from the AzureML workspace using the
        `get_mlflow_tracking_uri` method.
    """

    def tracking_uri(
        self,
        credentials: Dict[str, str]#,
        # options: Dict[str, str],
    ) -> str:
        """URI to use for tracking."""
        ws = Workspace(
            subscription_id=self.getkey(
                credentials, "subscription_id", "AZUREML_SUBSCRIPTION_ID"
            ),
            resource_group=self.getkey(
                credentials, "resource_group", "AZUREML_RESOURCE_GROUP"
            ),
            workspace_name=self.getkey(
                credentials, "workspace_name", "AZUREML_WORKSPACE_NAME"
            ),
        )
        return ws.get_mlflow_tracking_uri()


azureml_connection = AzureMLConnection()
