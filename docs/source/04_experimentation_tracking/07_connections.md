# Connecting to a mlflow server

## What is a connection?

Sometimes connecting to a tracking server may require more steps than just providing a URI. For example, if you want to connect to AzureML, part of the URI contains the location of the AzureML service you're using. In this case, it is recommended to get the URI from the AzureML Workspace object, and then connect it to the tracking server.

## How to use a connection?

To do so, you need to provide a connection keyword in the `tracking_uri` or `registry_uri`. This will tell `kedro-mlflow` to look for a connection plugin named `azureml` and will pass the credentials and connection options to it. For instance, if you want to connect to AzureML, you can do as follows:

```yaml
server:
  tracking_uri: azureml
  connection:
    subscription_id: my_subscription_id
    resource_group: my_resource_group
    workspace_name: my_workspace
```

## What connections are available?

Every connection may work differently when it comes to connection options or credentials. The official documentation of the builtin connections are available below:

### Databricks

Since `mlflow` already provides a treatment for URIs equal to `databricks`, you can use it directly in the `tracking_uri` or `registry_uri` keys. This connection doesn't calculate anything and will pass `databricks` as the URI to `mlflow`.

Example:

```yaml
server:
  tracking_uri: databricks
```

### AzureML

This connection uses the AzureML SDK v1 to calculate the URI from the Workspace object. It requires the following connection options:

- `subscription_id`: The subscription id of the AzureML workspace
- `resource_group`: The resource group of the AzureML workspace
- `workspace_name`: The name of the AzureML workspace

These options can also be specified using environment variables. In this case, the connection options will be respectively:

- `AZUREML_SUBSCRIPTION_ID`
- `AZUREML_RESOURCE_GROUP`
- `AZUREML_WORKSPACE_NAME`

Example:

```yaml
server:
  tracking_uri: azureml
  connection:
    subscription_id: my_subscription_id
    resource_group: my_resource_group
    workspace_name: my_workspace
```

## Custom connections

If the connection you want to use is not available, you can create your own connection plugin. To do so, you need to create a class that inherits from `kedro_mlflow.config.connection.KedroMlflowConnection` and implements at least the `tracking_uri` method.

Example:
```python
from kedro_mlflow.config.connection import KedroMlflowConnection


class MyConnection(KedroMlflowConnection):
    def tracking_uri(self, credentials, options) -> str:
        return "my_tracking_uri"


my_connection = MyConnection()
```

Then, you need to register one instance of your connection class as an entrypoint in your setup file pointing to `kedro_mlflow.connections`. For example, let's say you want to register `MyConnection` as a connection on a library implemented using `setuptools`. You can do as follows:

```python
from setuptools import setup

setup(
    ...,
    entry_points={
        "kedro_mlflow.connections": [
            "my_connection = my_package.my_module:my_connection"
        ]
    },
)
```

After that, you can use your connection in your `conf/base/mlflow.yml` file as follows:

```yaml
server:
  tracking_uri: my_connection
```
