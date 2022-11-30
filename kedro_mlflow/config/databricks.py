from kedro_mlflow.config.connection import KedroMlflowConnection


class DatabricksConnection(KedroMlflowConnection):
    """Parses `databricks` as itself because it is intended to be used as a keyword."""

    def tracking_uri(self, *_, **__) -> str:
        """URI to use for tracking."""
        return "databricks"


databricks_connection = DatabricksConnection()
