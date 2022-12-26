from kedro_mlflow.utils import _load_plugins, _parse_requirements, _flatten_dict, _unflatten_dict


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
def test_flatten_dict_non_nested():
    d = dict(a=1, b=2)
    assert _flatten_dict(d=d, recursive=True, sep=".") == d
    assert _flatten_dict(d=d, recursive=False, sep=".") == d


def test_flatten_dict_nested_1_level():
    d = dict(a=1, b=dict(c=3, d=4))
    flattened = {"a": 1, "b.c": 3, "b.d": 4}
    assert _flatten_dict(d=d, recursive=True, sep=".") == flattened
    assert _flatten_dict(d=d, recursive=False, sep=".") == flattened


def test_flatten_dict_nested_2_levels():
    d = dict(a=1, b=dict(c=1, d=dict(e=3, f=5)))

    assert _flatten_dict(d=d, recursive=True, sep=".") == {
        "a": 1,
        "b.c": 1,
        "b.d.e": 3,
        "b.d.f": 5,
    }
    assert _flatten_dict(d=d, recursive=False, sep=".") == {
        "a": 1,
        "b.c": 1,
        "b.d": {"e": 3, "f": 5},
    }


def test_flatten_dict_nested_3_levels():
    d = dict(a=1, b=dict(c=1, d=dict(e=3, f=dict(g=4, h=5))))

    assert _flatten_dict(d=d, recursive=True, sep=".") == {
        "a": 1,
        "b.c": 1,
        "b.d.e": 3,
        "b.d.f.g": 4,
        "b.d.f.h": 5,
    }
    assert _flatten_dict(d=d, recursive=False, sep=".") == {
        "a": 1,
        "b.c": 1,
        "b.d": {"e": 3, "f": {"g": 4, "h": 5}},
    }


def test_flatten_dict_with_float_keys():
    d = {0: 1, 1: {3: 1, 4: {"e": 3, 6.7: 5}}}

    assert _flatten_dict(d=d, recursive=True, sep="_") == {
        "0": 1,
        "1_3": 1,
        "1_4_e": 3,
        "1_4_6.7": 5,
    }
    assert _flatten_dict(d=d, recursive=False, sep="_") == {
        "0": 1,
        "1_3": 1,
        "1_4": {
            "e": 3,
            6.7: 5,  # 6.7 is not converted to string, but when the entire dict will be logged mlflow will take care of the conversion
        },
    }


def test_flatten_dict_with_user_defined_sep():
    d = dict(a=1, b=dict(c=1, d=dict(e=3, f=dict(g=4, h=5))))

    assert _flatten_dict(d=d, recursive=True, sep="_") == {
        "a": 1,
        "b_c": 1,
        "b_d_e": 3,
        "b_d_f_g": 4,
        "b_d_f_h": 5,
    }


def test_unflatten_dict_non_nested():
    d = dict(a=1, b=2)
    assert _unflatten_dict(d=d, sep=".") == d


def test_unflatten_dict_nested_1_level():
    d = {"a": 1, "b.c": 3, "b.d": 4}
    u = _unflatten_dict(d=d, sep=".")

    assert u["a"] == 1
    assert u["b"]["c"] == 3
    assert u["b"]["d"] == 4


def test_unflatten_dict_nested_2_levels():
    d = {"a": 1, "b.c": 1, "b.d.e": 3, "b.d.f": 5}
    u = _unflatten_dict(d=d, sep=".")

    assert u["a"] == 1
    assert u["b"]["c"] == 1
    assert u["b"]["d"]["e"] == 3
    assert u["b"]["d"]["f"] == 5


def test_unflatten_dict_nested_3_levels():
    d = {
        "a": 1,
        "b_c": 1,
        "b_d_e": 3,
        "b_d_f_g": 4,
        "b_d_f_h": 5,
    }
    u = _unflatten_dict(d=d, sep="_")

    assert u["a"] == 1
    assert u["b"]["c"] == 1
    assert u["b"]["d"]["e"] == 3
    assert u["b"]["d"]["f"]["g"] == 4
    assert u["b"]["d"]["f"]["h"] == 5


def test_unflatten_dict_with_user_defined_sep():
    d = {
        "a": 1,
        "b_c": 1,
        "b_d_e": 3,
        "b_d_f_g": 4,
        "b_d_f_h": 5,
    }
    u = _unflatten_dict(d=d, sep="_")

    assert u["a"] == 1
    assert u["b"]["c"] == 1
    assert u["b"]["d"]["e"] == 3
    assert u["b"]["d"]["f"]["g"] == 4
    assert u["b"]["d"]["f"]["h"] == 5
