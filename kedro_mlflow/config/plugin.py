import abc
from typing import Dict


class KedroMlflowConnection(abc.ABC):
    """Allows generating custom connection setup for the KedroMlflow plugin.

    This is useful for example when you want to use a service that generates
    dynamic URLs for the server or when the logic to generate the URL is
    complex.

    To use a connection, the user must specify the name of the plugin entrypoint
    in an uri key in the `mflow.yml` file. For example, if the entrypoint is
    `databricks`, the user will input `databricks` in the `mlflow_tracking_uri`
    setting.
    """

    @abc.abstractmethod
    def tracking_uri(self, credentials: Dict[str, str] = None) -> str:
        """URI to use for tracking."""
        pass  # pragma: no cover

    def registry_uri(self, credentials: Dict[str, str] = None) -> str:
        """URI to use for registry."""
        return self.tracking_uri()
