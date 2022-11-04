import pytest

from kedro_mlflow.config.plugin import KedroMlflowConnection


class FakeConnection(KedroMlflowConnection):
    def tracking_uri(self, credentials: dict = None) -> str:
        """URI to use for tracking."""
        return "fake"


@pytest.fixture
def fake_connection():
    return FakeConnection()


def test_connection_tracking_uri(fake_connection):
    assert fake_connection.tracking_uri() == "fake"


def test_connection_registry_uri(fake_connection):
    assert fake_connection.registry_uri() == "fake"
