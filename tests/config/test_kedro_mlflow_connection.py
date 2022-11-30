import pytest

from kedro_mlflow.config.connection import KedroMlflowConnection


class FakeConnection(KedroMlflowConnection):
    def tracking_uri(self, credentials: dict, options: dict) -> str:
        """URI to use for tracking."""
        return "fake"


@pytest.fixture
def fake_connection():
    return FakeConnection()


def test_connection_tracking_uri(fake_connection):
    assert fake_connection.tracking_uri({}, {}) == "fake"


def test_connection_registry_uri(fake_connection):
    assert fake_connection.registry_uri({}, {}) == "fake"


def test_connection_getkey_sane(fake_connection):
    assert fake_connection.getkey({"key": "value"}, "key", "KEY") == "value"


def test_connection_getkey_no_key(fake_connection):
    with pytest.raises(KeyError):
        fake_connection.getkey({}, "key", "KEY")


def test_connection_getkey_env(fake_connection, mocker):
    mocker.patch("os.environ", {"KEY": "value"})
    assert fake_connection.getkey({"key": "val"}, "key", "KEY") == "val"
    assert fake_connection.getkey({}, "key", "KEY") == "value"


def test_connection_getkey_default(fake_connection, mocker):
    assert fake_connection.getkey({}, "key", "KEY", "default") == "default"
    mocker.patch("os.environ", {"KEY": "value"})
    assert fake_connection.getkey({}, "key", "KEY", "default") == "value"
    assert fake_connection.getkey({"key": "val"}, "key", "default") == "val"
