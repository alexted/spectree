from typing import Annotated, get_origin

import pytest

from spectree.model_adapter import ModelSpec
from spectree.utils import get_model_key
from tests.common_dataclass import DemoModel, SimpleModel


def _partial_model_instance_value(model_case, kind):
    simple_model = model_case.get_model(SimpleModel)
    values = {
        "model": simple_model(user_id=1),
        "list-with-model": [0, simple_model(user_id=1)],
        "list-without-model": [1, 2, 3],
        "tuple-with-model": (0, simple_model(user_id=1)),
        "tuple-without-model": (0, 1),
        "dict-with-model": {"test": simple_model(user_id=1)},
        "nested-dict-with-model": {"test": [simple_model(user_id=1)]},
        "nested-list-with-model": [0, [1, simple_model(user_id=1)]],
    }

    return values[kind]


def test_validate_obj_and_is_model_instance(model_case):
    adapter = model_case.adapter
    simple_model = model_case.get_model(SimpleModel)

    instance = model_case.validate_obj(simple_model, {"user_id": "1"})

    assert model_case.dump_python(instance) == {"user_id": 1}
    assert adapter.is_model_instance(instance, simple_model) is True
    assert adapter.is_model_instance({"user_id": 1}, simple_model) is False
    assert adapter.is_model_instance(simple_model, simple_model) is False


def test_generic_list_model_spec(model_case):
    adapter = model_case.adapter

    model = model_case.get_model(SimpleModel)

    spec = list[model]

    assert adapter.is_model_type(spec) is True

    value = adapter.validate_obj(
        spec,
        [{"user_id": "1"}, {"user_id": "2"}],
    )

    assert len(value) == 2
    assert all(adapter.is_model_instance(item, model) for item in value)

    assert (
        adapter.is_model_instance(
            [{"user_id": 1}],
            spec,
        )
        is False
    )

    decoded = adapter.validate_json(
        spec,
        b'[{"user_id": 1}, {"user_id": 2}]',
    )

    assert len(decoded) == 2


def test_generic_dict_model_spec(model_case):
    adapter = model_case.adapter

    model = model_case.get_model(SimpleModel)

    spec = dict[str, model]

    assert adapter.is_model_type(spec) is True

    value = adapter.validate_obj(
        spec,
        {
            "first": {"user_id": "1"},
            "second": {"user_id": "2"},
        },
    )

    assert all(adapter.is_model_instance(item, model) for item in value.values())

    assert (
        adapter.is_model_instance(
            value,
            spec,
        )
        is True
    )

    assert (
        adapter.is_model_instance(
            {
                "first": {"user_id": 1},
            },
            spec,
        )
        is False
    )


def test_annotated_model_spec(model_case):
    adapter = model_case.adapter

    model = model_case.get_model(SimpleModel)

    spec = Annotated[
        model,
        "spectree-test-metadata",
    ]

    assert adapter.is_model_type(spec) is True

    instance = adapter.validate_obj(
        spec,
        {"user_id": "1"},
    )

    assert (
        adapter.is_model_instance(
            instance,
            spec,
        )
        is True
    )

    assert adapter.dump_json(instance) == b'{"user_id":1}'


def test_nested_generic_model_spec(model_case):
    adapter = model_case.adapter

    model = model_case.get_model(SimpleModel)

    spec = list[dict[str, model]]

    value = adapter.validate_obj(
        spec,
        [
            {
                "first": {"user_id": "1"},
                "second": {"user_id": "2"},
            }
        ],
    )

    assert (
        adapter.is_model_instance(
            value,
            spec,
        )
        is True
    )

    assert (
        adapter.is_model_instance(
            [
                {
                    "first": {"user_id": 1},
                }
            ],
            spec,
        )
        is False
    )


def test_optional_model_spec(model_case):
    adapter = model_case.adapter

    model = model_case.get_model(SimpleModel)
    spec = model | None

    assert adapter.is_model_type(spec) is True

    instance = adapter.validate_obj(
        spec,
        {"user_id": "1"},
    )

    assert (
        adapter.is_model_instance(
            instance,
            spec,
        )
        is True
    )

    assert (
        adapter.is_model_instance(
            None,
            spec,
        )
        is True
    )


def test_converted_model_is_supported_as_a_model(model_case):
    adapter = model_case.adapter

    instance = adapter.validate_obj(SimpleModel, {"user_id": "1"})

    assert instance == SimpleModel(user_id=1)
    assert adapter.is_model_type(SimpleModel) is True
    assert adapter.is_model_instance(instance, SimpleModel) is True
    assert adapter.dump_json(instance) == b'{"user_id":1}'
    assert (
        adapter.json_schema(
            SimpleModel,
            ref_template="#/components/schemas/{model}",
        )["properties"]["user_id"]["type"]
        == "integer"
    )


def test_root_model_instances(model_case):
    adapter = model_case.adapter
    dummy_root_model = model_case.get_model(list[int], name="DummyRootModel")
    nested_root_model = model_case.get_model(
        dummy_root_model,
        name="NestedRootModel",
    )

    root_instance = model_case.validate_obj(
        dummy_root_model,
        [1, 2, 3],
    )
    nested_root_instance = model_case.validate_obj(
        nested_root_model,
        root_instance,
    )

    assert adapter.is_model_instance(root_instance, dummy_root_model) is True
    assert (
        adapter.is_model_instance(
            nested_root_instance,
            nested_root_model,
        )
        is True
    )
    assert model_case.dump_python(nested_root_instance) == [1, 2, 3]


def test_plain_dataclass_with_root_like_field_is_model_instance(model_case):
    lookalike = model_case.root_model_lookalike(__root__=["False"])

    assert (
        model_case.adapter.is_model_instance(
            lookalike,
            model_case.root_model_lookalike,
        )
        is True
    )


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("model", True),
        ("list-with-model", True),
        ("list-without-model", False),
        ("tuple-with-model", True),
        ("tuple-without-model", False),
        ("dict-with-model", True),
        ("nested-dict-with-model", True),
        ("nested-list-with-model", True),
    ],
)
def test_is_partial_model_instance(model_case, kind, expected):
    value = _partial_model_instance_value(model_case, kind)

    assert model_case.adapter.is_partial_model_instance(value) is expected


def test_dump_json(model_case):
    simple_model = model_case.get_model(SimpleModel)

    assert model_case.dump_python(simple_model(user_id=1)) == {"user_id": 1}
    assert model_case.dump_python(
        model_case.validate_obj(
            model_case.get_model(list[int], name="DummyRootModel"),
            [1, 2, 3],
        )
    ) == [1, 2, 3]
    assert model_case.dump_python(
        model_case.validate_obj(
            model_case.get_model(list[SimpleModel], name="Users"),
            [
                {"user_id": 1},
                {"user_id": 2},
            ],
        )
    ) == [{"user_id": 1}, {"user_id": 2}]


def test_validate_json_list_model(model_case):
    list_model = model_case.adapter.make_list_model(model_case.get_model(SimpleModel))
    instance = model_case.validate_json(list_model, b'[{"user_id": 1}, {"user_id": 2}]')

    assert model_case.dump_python(instance) == [
        {"user_id": 1},
        {"user_id": 2},
    ]


def test_json_schema(model_case):
    schema = model_case.adapter.json_schema(
        model_case.get_model(SimpleModel),
        ref_template="#/components/schemas/{model}",
    )

    assert schema["type"] == "object"
    assert schema["properties"]["user_id"]["type"] == "integer"


def test_validation_errors(model_case):
    with pytest.raises(model_case.adapter.validation_error) as exc_info:
        model_case.validate_obj(model_case.get_model(SimpleModel), {"user_id": "bad"})

    errors = model_case.adapter.validation_errors(exc_info.value)

    assert isinstance(errors, list)
    assert list(errors[0]["loc"]) == ["user_id"]
    assert errors[0]["msg"]
    assert errors[0]["type"]


def test_model_spec_accepts_generic_alias():
    model: ModelSpec = list[DemoModel]
    assert model == list[DemoModel]


def test_generic_model_json_roundtrip(model_case):
    adapter = model_case.adapter

    model = model_case.get_model(SimpleModel)

    spec = list[model]

    original = [
        model(user_id=1),
        model(user_id=2),
    ]

    payload = adapter.dump_json(original)

    decoded = adapter.validate_json(
        spec,
        payload,
    )

    assert adapter.is_model_instance(
        decoded,
        spec,
    )

    assert model_case.dump_python(decoded) == [
        {"user_id": 1},
        {"user_id": 2},
    ]


def test_generic_model_schema(model_case):
    adapter = model_case.adapter

    model = model_case.get_model(SimpleModel)

    schema = adapter.json_schema(
        list[model],
        ref_template="#/components/schemas/{model}",
    )

    assert schema["type"] == "array"


def test_annotated_generic_model_schema(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)

    schema = adapter.json_schema(
        Annotated[list[model], "metadata"],
        ref_template="#/components/schemas/{model}",
    )

    assert schema["type"] == "array"


def test_model_key_is_stable_for_generic_model_spec():
    model = SimpleModel

    first = get_model_key(list[model])
    second = get_model_key(list[model])

    assert first == second


def test_model_key_distinguishes_generic_model_shapes():
    model = SimpleModel

    assert get_model_key(list[model]) != get_model_key(dict[str, model])
    assert get_model_key(list[model]) != get_model_key(model | None)


def test_model_key_preserves_annotated_title():
    model = SimpleModel

    plain = get_model_key(model)
    titled = get_model_key(
        Annotated[
            model,
            type("Metadata", (), {"title": "CustomUser"})(),
        ]
    )

    assert plain != titled
    assert titled.startswith("CustomUser.")


@pytest.mark.parametrize(
    "spec_factory",
    [
        lambda model: list[model],
        lambda model: dict[str, model],
        lambda model: model | None,
        lambda model: Annotated[model, "metadata"],
    ],
)
def test_model_spec_json_roundtrip(model_case, spec_factory):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)
    spec = spec_factory(model)

    assert adapter.is_model_type(spec)

    value = adapter.validate_obj(
        spec,
        (
            [{"user_id": 1}]
            if get_origin(spec) is list
            else {"item": {"user_id": 1}}
            if get_origin(spec) is dict
            else {"user_id": 1}
        ),
    )

    payload = adapter.dump_json(value)
    restored = adapter.validate_json(spec, payload)

    assert adapter.is_model_instance(
        restored,
        spec,
    )


def test_model_key_is_deterministic_for_object_metadata():
    class Metadata:
        def __init__(self, value: str):
            self.value = value

    first = get_model_key(
        Annotated[
            SimpleModel,
            Metadata("test"),
        ]
    )
    second = get_model_key(
        Annotated[
            SimpleModel,
            Metadata("test"),
        ]
    )

    assert first == second


def test_model_key_changes_when_annotated_metadata_changes():
    class Metadata:
        def __init__(self, value: str):
            self.value = value

    first = get_model_key(
        Annotated[
            SimpleModel,
            Metadata("first"),
        ]
    )
    second = get_model_key(
        Annotated[
            SimpleModel,
            Metadata("second"),
        ]
    )

    assert first != second


def test_model_key_is_deterministic_for_nested_annotated_generic():
    spec = Annotated[
        list[dict[str, SimpleModel]],
        "metadata",
    ]

    assert get_model_key(spec) == get_model_key(spec)
