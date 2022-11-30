import abc
import os
from typing import Any, Dict


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

    @staticmethod
    def getkey(mapping: dict, key: str, envkey: str, default: Any = None) -> Any:
        """Get a key from a dictionary or an environment variable.

        Args:
            mapping (dict): Dictionary to get the key from.
            key (str): Key to get.
            envkey (str): Environment variable to get the key from.
            default (Optional[Any]): Default value to return if the key is not found.

        Returns:
            Any: Value of the key.

        Raises:
            KeyError: If the key is not found and no default value is provided.
        """
        ret = mapping.get(key, os.environ.get(envkey, default))
        if ret is None:
            raise KeyError(
                f"Key '{key}' not found in specified credentials nor in '{envkey}' "
                "environment variable."
            )
        return ret

    @abc.abstractmethod
    def tracking_uri(
        self,
        credentials: Dict[str, str],
        options: Dict[str, str],
    ) -> str:
        """URI to use for tracking."""

    def registry_uri(
        self,
        credentials: Dict[str, str],
        options: Dict[str, str],
    ) -> str:
        """URI to use for registry."""
        return self.tracking_uri(credentials, options)
