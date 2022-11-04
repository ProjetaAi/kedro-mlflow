from kedro_mlflow.config.plugin import KedroMlflowConnection


class DatabricksConnection(KedroMlflowConnection):
    """Parses `databricks` as itself because it is intended to be used as a keyword."""

    def tracking_uri(self, credentials: dict = None) -> str:
        """URI to use for tracking."""
        return "databricks"


databricks_connection = DatabricksConnection()
