from kedro_mlflow.config.databricks import DatabricksConnection


def test_connection_tracking_uri():
    assert DatabricksConnection().tracking_uri({}, {}) == "databricks"


def test_connection_registry_uri():
    assert DatabricksConnection().registry_uri({}, {}) == "databricks"
