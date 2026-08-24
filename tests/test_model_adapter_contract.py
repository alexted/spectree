from typing import Annotated, Literal

import pytest

from spectree.utils import get_model_key, hash_module_path
from tests.common_dataclass import (
    NestedDataclass,
    SimpleModel,
    DemoModel
)
from spectree.model_adapter import ModelSpec



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

    instance = model_case.validate_obj(
        simple_model,
        {"user_id": "1"},
    )

    assert model_case.dump_python(instance) == {"user_id": 1}
    assert adapter.is_model_instance(instance, simple_model) is True
    assert adapter.is_model_instance({"user_id": 1}, simple_model) is False
    assert adapter.is_model_instance(simple_model, simple_model) is False


@pytest.mark.parametrize("wrapper", [list, tuple])
def test_compiled_model_preserves_nested_instances(
    model_case,
    wrapper,
):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)
    spec = wrapper[model]

    value = [{"user_id": 1}] if wrapper is list else ({"user_id": 1},)

    instance = adapter.validate_obj(spec, value)

    expected = [model(user_id=1)] if wrapper is list else (model(user_id=1),)

    assert instance == expected
    assert adapter.is_model_instance(instance, spec) is True
    assert adapter.is_model_instance(value, spec) is False


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
            [model(user_id=1)],
            spec,
        )
        is True
    )

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
            {
                "first": model(user_id=1),
                "second": model(user_id=2),
            },
            spec,
        )
        is True
    )

    assert (
        adapter.is_model_instance(
            {
                "first": {"user_id": 1},
                "second": {"user_id": 2},
            },
            spec,
        )
        is False
    )


def test_annotated_model_spec(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)
    spec = Annotated[model, "metadata"]

    assert adapter.is_model_type(spec) is True

    instance = adapter.validate_obj(
        spec,
        {"user_id": "1"},
    )

    assert adapter.is_model_instance(instance, spec) is True
    assert adapter.is_model_instance({"user_id": 1}, spec) is False


def test_optional_model_spec(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)
    spec = model | None

    assert adapter.is_model_type(spec) is True

    instance = adapter.validate_obj(
        spec,
        {"user_id": "1"},
    )

    assert adapter.is_model_instance(instance, spec) is True
    assert adapter.is_model_instance(None, spec) is True
    assert adapter.is_model_instance({"user_id": 1}, spec) is False


def test_literal_model_spec(model_case):
    adapter = model_case.adapter
    spec = Literal["ok", "accepted"]

    assert adapter.is_model_type(spec) is True
    assert adapter.is_model_instance("ok", spec) is True
    assert adapter.is_model_instance("accepted", spec) is True
    assert adapter.is_model_instance("rejected", spec) is False


def test_nested_generic_model_spec(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)
    spec = list[dict[str, model | None]]

    valid = [
        {
            "first": model(user_id=1),
            "second": None,
        }
    ]
    invalid = [
        {
            "first": {"user_id": 1},
            "second": None,
        }
    ]

    assert adapter.is_model_instance(valid, spec) is True
    assert adapter.is_model_instance(invalid, spec) is False


def test_compiled_model_is_cached(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)

    first = adapter.compile(model)
    second = adapter.compile(model)

    assert first is second


def test_compiled_model_owns_runtime_validation(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)

    compiled = adapter.compile(model)

    with pytest.raises(adapter.validation_error):
        compiled.validate_obj({"user_id": "bad"})


def test_compiled_model_json(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)

    compiled = adapter.compile(model)

    instance = compiled.validate_json(b'{"user_id": 1}')

    assert model_case.dump_python(instance) == {"user_id": 1}


def test_compiled_model_schema(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)

    compiled = adapter.compile(model)

    schema = compiled.json_schema(
        ref_template="#/components/schemas/{model}",
    )

    assert schema["type"] == "object"
    assert schema["properties"]["user_id"]["type"] == "integer"


def test_compiled_generic_model(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(
        list[SimpleModel],
        name="Users",
    )

    compiled = adapter.compile(model)

    instance = compiled.validate_obj(
        [
            {"user_id": 1},
            {"user_id": 2},
        ],
    )

    assert model_case.dump_python(instance) == [
        {"user_id": 1},
        {"user_id": 2},
    ]


def test_compiled_model_preserves_dataclass_identity(model_case):
    adapter = model_case.adapter
    model = SimpleModel
    instance = model(user_id=1)

    compiled = adapter.compile(model)

    assert compiled.is_instance(instance) is True
    assert compiled.is_instance({"user_id": 1}) is False

    validated = compiled.validate_obj(instance)

    assert validated is instance
    assert type(validated) is model


def test_plain_dataclass_is_supported_as_model(model_case):
    adapter = model_case.adapter

    instance = adapter.validate_obj(
        SimpleModel,
        {"user_id": "1"},
    )

    assert instance == SimpleModel(user_id=1)
    assert type(instance) is SimpleModel
    assert adapter.is_model_type(SimpleModel) is True
    assert adapter.is_model_instance(instance, SimpleModel) is True
    assert (
        adapter.is_model_instance(
            {"user_id": 1},
            SimpleModel,
        )
        is False
    )


def test_nested_dataclass_is_supported_as_model(model_case):
    adapter = model_case.adapter

    instance = adapter.validate_obj(
        NestedDataclass,
        {
            "child": {
                "user_id": "1",
            },
        },
    )

    assert type(instance) is NestedDataclass
    assert instance.child == SimpleModel(user_id=1)


def test_dataclass_list_is_supported_as_model(model_case):
    adapter = model_case.adapter
    model = list[SimpleModel]

    instance = adapter.validate_obj(
        model,
        [
            {"user_id": 1},
            {"user_id": 2},
        ],
    )

    assert instance == [
        SimpleModel(user_id=1),
        SimpleModel(user_id=2),
    ]
    assert all(type(item) is SimpleModel for item in instance)
    assert adapter.is_model_instance(instance, model) is True


def test_annotated_dataclass_is_supported_as_model(model_case):
    adapter = model_case.adapter
    model = Annotated[SimpleModel, "metadata"]

    instance = adapter.validate_obj(
        model,
        {"user_id": 1},
    )

    assert instance == SimpleModel(user_id=1)
    assert type(instance) is SimpleModel
    assert adapter.is_model_instance(instance, model) is True


def test_dataclass_json_roundtrip(model_case):
    adapter = model_case.adapter
    original = SimpleModel(user_id=42)

    encoded = adapter.dump_json(original)
    decoded = adapter.validate_json(
        SimpleModel,
        encoded,
    )

    assert decoded == original
    assert type(decoded) is SimpleModel


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
def test_partial_model_instance_contract(
    model_case,
    kind,
    expected,
):
    value = _partial_model_instance_value(model_case, kind)

    assert model_case.adapter.is_partial_model_instance(value) is expected


def test_dump_json(model_case):
    simple_model = model_case.get_model(SimpleModel)

    assert model_case.dump_python(
        simple_model(user_id=1),
    ) == {"user_id": 1}

    assert model_case.dump_python(
        model_case.validate_obj(
            model_case.get_model(
                list[int],
                name="DummyRootModel",
            ),
            [1, 2, 3],
        ),
    ) == [1, 2, 3]

    assert model_case.dump_python(
        model_case.validate_obj(
            model_case.get_model(
                list[SimpleModel],
                name="Users",
            ),
            [
                {"user_id": 1},
                {"user_id": 2},
            ],
        ),
    ) == [
        {"user_id": 1},
        {"user_id": 2},
    ]


def test_validate_json_list_model(model_case):
    list_model = model_case.adapter.make_list_model(
        model_case.get_model(SimpleModel),
    )

    instance = model_case.validate_json(
        list_model,
        b'[{"user_id": 1}, {"user_id": 2}]',
    )

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
        model_case.validate_obj(
            model_case.get_model(SimpleModel),
            {"user_id": "bad"},
        )

    errors = model_case.adapter.validation_errors(exc_info.value)

    assert errors
    assert all("loc" in error for error in errors)


def test_model_key_changes_when_annotated_metadata_changes():
    class Metadata:
        def __init__(self, value: str):
            self.value = value

    first = get_model_key(
        Annotated[
            SimpleModel,
            Metadata("first"),
        ],
    )

    second = get_model_key(
        Annotated[
            SimpleModel,
            Metadata("second"),
        ],
    )

    assert first != second


def test_model_key_is_deterministic_for_nested_annotated_generic():
    spec = Annotated[
        list[dict[str, SimpleModel]],
        "metadata",
    ]

    assert get_model_key(spec) == get_model_key(spec)


def test_plain_model_key_is_backward_compatible():
    module_hash = hash_module_path(
        module_path=SimpleModel.__module__,
    )

    assert get_model_key(SimpleModel) == (f"SimpleModel.{module_hash}")


def test_generic_model_key_is_deterministic():
    spec = list[SimpleModel]

    assert get_model_key(spec) == get_model_key(spec)


def test_compiled_model_dump_json(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)
    compiled = adapter.compile(model)
    instance = compiled.validate_obj({"user_id": 1})

    payload = compiled.dump_json(instance)

    assert payload == b'{"user_id":1}'


def test_compiled_model_annotation_preserves_origin(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)
    spec = Annotated[list[model], "metadata"]

    compiled = adapter.compile(spec)
    instance = compiled.validate_obj(
        [
            {"user_id": 1},
            {"user_id": 2},
        ],
    )

    assert all(type(item) is model for item in instance)
    assert compiled.is_instance(instance) is True
    assert (
        compiled.is_instance(
            [
                {"user_id": 1},
                {"user_id": 2},
            ],
        )
        is False
    )


def test_model_spec_accepts_generic_alias():
    model: ModelSpec = list[DemoModel]
    assert model == list[DemoModel]


def test_compiled_model(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)

    compiled = adapter.compile(model)

    instance = compiled.validate_obj({"user_id": "1"})

    assert model_case.dump_python(instance) == {"user_id": 1}
    assert compiled.is_instance(instance) is True
    assert compiled.is_instance({"user_id": 1}) is False


def test_compiled_model_json(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)

    compiled = adapter.compile(model)

    instance = compiled.validate_json(b'{"user_id": 1}')

    assert model_case.dump_python(instance) == {"user_id": 1}


def test_compiled_model_schema(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(SimpleModel)

    compiled = adapter.compile(model)

    schema = compiled.json_schema(
        ref_template="#/components/schemas/{model}",
    )

    assert schema["type"] == "object"
    assert schema["properties"]["user_id"]["type"] == "integer"


def test_compiled_generic_model(model_case):
    adapter = model_case.adapter
    model = model_case.get_model(
        list[SimpleModel],
        name="Users",
    )

    compiled = adapter.compile(model)

    instance = compiled.validate_obj(
        [
            {"user_id": 1},
            {"user_id": 2},
        ]
    )

    assert model_case.dump_python(instance) == [
        {"user_id": 1},
        {"user_id": 2},
    ]

