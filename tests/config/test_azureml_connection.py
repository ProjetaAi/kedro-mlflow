import pytest

from kedro_mlflow.config.azureml import AzureMLConnection


class TestWorkspace:
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        *_,
        **__,
    ):
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name

    def get_mlflow_tracking_uri(self):
        return f"{self._subscription_id}/{self._resource_group}/{self._workspace_name}"


@pytest.fixture(autouse=True)
def mock_ws(mocker):
    mocker.patch("kedro_mlflow.config.azureml.Workspace", TestWorkspace)


def test_azureml_tracking_uri_missing_options():
    with pytest.raises(KeyError):
        AzureMLConnection().tracking_uri({}, {})

    with pytest.raises(KeyError):
        AzureMLConnection().tracking_uri(
            {}, {"subscription_id": "1234", "resource_group": "1234"}
        )

    with pytest.raises(KeyError):
        AzureMLConnection().tracking_uri(
            {}, {"resource_group": "1234", "workspace_name": "1234"}
        )

    with pytest.raises(KeyError):
        AzureMLConnection().tracking_uri(
            {}, {"workspace_name": "1234", "subscription_id": "1234"}
        )


def test_azureml_registry_uri_missing_options():
    with pytest.raises(KeyError):
        AzureMLConnection().registry_uri({}, {})

    with pytest.raises(KeyError):
        AzureMLConnection().registry_uri(
            {}, {"subscription_id": "1234", "resource_group": "1234"}
        )

    with pytest.raises(KeyError):
        AzureMLConnection().registry_uri(
            {}, {"resource_group": "1234", "workspace_name": "1234"}
        )

    with pytest.raises(KeyError):
        AzureMLConnection().registry_uri(
            {}, {"workspace_name": "1234", "subscription_id": "1234"}
        )


def test_azureml_tracking_uri_sane():
    assert (
        AzureMLConnection().tracking_uri(
            {}, {"subscription_id": "a", "resource_group": "b", "workspace_name": "c"}
        )
        == "a/b/c"
    )


def test_azureml_registry_uri_sane():
    assert (
        AzureMLConnection().registry_uri(
            {}, {"subscription_id": "a", "resource_group": "b", "workspace_name": "c"}
        )
        == "a/b/c"
    )


def test_azureml_tracking_uri_env(mocker):
    mocker.patch(
        "os.environ",
        {
            "AZUREML_SUBSCRIPTION_ID": "a",
            "AZUREML_RESOURCE_GROUP": "b",
            "AZUREML_WORKSPACE_NAME": "c",
        },
    )
    assert AzureMLConnection().tracking_uri({}, {}) == "a/b/c"
    assert (
        AzureMLConnection().tracking_uri({}, {"subscription_id": "1234"}) == "1234/b/c"
    )
    assert (
        AzureMLConnection().tracking_uri({}, {"resource_group": "1234"}) == "a/1234/c"
    )
    assert (
        AzureMLConnection().tracking_uri({}, {"workspace_name": "1234"}) == "a/b/1234"
    )


def test_azureml_registry_uri_env(mocker):
    mocker.patch(
        "os.environ",
        {
            "AZUREML_SUBSCRIPTION_ID": "a",
            "AZUREML_RESOURCE_GROUP": "b",
            "AZUREML_WORKSPACE_NAME": "c",
        },
    )
    assert AzureMLConnection().registry_uri({}, {}) == "a/b/c"
    assert (
        AzureMLConnection().registry_uri({}, {"subscription_id": "1234"}) == "1234/b/c"
    )
    assert (
        AzureMLConnection().registry_uri({}, {"resource_group": "1234"}) == "a/1234/c"
    )
    assert (
        AzureMLConnection().registry_uri({}, {"workspace_name": "1234"}) == "a/b/1234"
    )
