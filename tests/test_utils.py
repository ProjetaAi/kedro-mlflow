from kedro_mlflow.utils import _load_plugins, _parse_requirements


def test_parse_requirements(tmp_path):

    with open(tmp_path / "requirements.txt", "w") as f:
        f.writelines(["kedro==0.17.0\n", " mlflow==1.11.0\n" "-r pandas\n"])

    requirements = _parse_requirements(tmp_path / "requirements.txt")
    expected_requirements = ["kedro==0.17.0", "mlflow==1.11.0"]

    assert requirements == expected_requirements


class FakeEntryPoint:
    def __init__(self, name="fake"):
        self.name = name

    def load(self):
        return "here"


def test_load_plugins(mocker):
    mocker.patch(
        "importlib.metadata.entry_points",
        return_value={
            "test": [FakeEntryPoint(name="fake1"), FakeEntryPoint(name="fake2")]
        },
    )
    plugins = _load_plugins("test")
    assert plugins.keys() == {"fake1", "fake2"}
    assert plugins["fake1"]() == "here"
