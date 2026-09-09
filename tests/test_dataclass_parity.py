from dataclasses import dataclass
from typing import Annotated

import pytest

from tests.common_dataclass import NestedDataclass, SimpleModel


@dataclass(frozen=True, slots=True)
class FrozenSlotsDataclass:
    user_id: int
    label: str = "default"


def test_slots_and_frozen_dataclass_full_adapter_contract(model_case):
    adapter = model_case.adapter
    model = FrozenSlotsDataclass
    instance = model(user_id=1)

    assert adapter.is_model_type(model) is True
    assert adapter.is_model_instance(instance, model) is True
    assert adapter.is_model_instance({"user_id": 1}, model) is False

    validated = adapter.validate_obj(
        model,
        {"user_id": "1"},
    )
    assert type(validated) is model
    assert validated == model(user_id=1)

    decoded = adapter.validate_json(
        model,
        b'{"user_id":1}',
    )
    assert type(decoded) is model
    assert decoded == model(user_id=1)

    assert adapter.dump_json(instance) == b'{"user_id":1,"label":"default"}'

    compiled = adapter.compile(model)
    assert compiled.model_spec is model
    assert compiled.is_instance(instance) is True
    assert compiled.validate_obj(instance) is instance
    assert compiled.validate_json(b'{"user_id":1}') == instance
    assert compiled.dump_json(instance) == adapter.dump_json(instance)

    schema = compiled.json_schema(
        ref_template="#/components/schemas/{model}",
    )
    assert schema["type"] == "object"
    assert schema["properties"]["user_id"]["type"] == "integer"
    assert schema["properties"]["label"]["default"] == "default"


def test_nested_dataclass_json_roundtrip_preserves_original_types(model_case):
    adapter = model_case.adapter
    payload = b'{"child":{"user_id":42},"tags":[1,2,3]}'

    decoded = adapter.validate_json(NestedDataclass, payload)

    assert type(decoded) is NestedDataclass
    assert type(decoded.child) is SimpleModel
    assert decoded.child == SimpleModel(user_id=42)
    assert decoded.tags == [1, 2, 3]
    assert adapter.dump_json(decoded) == payload


def test_annotated_dataclass_json_roundtrip_preserves_original_type(model_case):
    adapter = model_case.adapter
    model = Annotated[SimpleModel, "metadata"]

    decoded = adapter.validate_json(
        model,
        b'{"user_id":7}',
    )

    assert type(decoded) is SimpleModel
    assert decoded == SimpleModel(user_id=7)
    assert adapter.is_model_instance(decoded, model) is True
    assert adapter.dump_json(decoded) == b'{"user_id":7}'


def test_list_dataclass_json_roundtrip_preserves_item_types(model_case):
    adapter = model_case.adapter
    model = list[SimpleModel]

    decoded = adapter.validate_json(
        model,
        b'[{"user_id":1},{"user_id":2}]',
    )

    assert [type(item) for item in decoded] == [SimpleModel, SimpleModel]
    assert decoded == [
        SimpleModel(user_id=1),
        SimpleModel(user_id=2),
    ]
    assert adapter.is_model_instance(decoded, model) is True
    assert adapter.dump_json(decoded) == b'[{"user_id":1},{"user_id":2}]'


@pytest.mark.parametrize(
    "model",
    [
        SimpleModel | None,
        list[SimpleModel],
        dict[str, SimpleModel],
        Annotated[SimpleModel, "metadata"],
    ],
)
def test_dataclass_generic_specs_preserve_instances_after_json_validation(
    model_case,
    model,
):
    adapter = model_case.adapter

    payloads = {
        SimpleModel | None: b'{"user_id":1}',
        list[SimpleModel]: b'[{"user_id":1}]',
        dict[str, SimpleModel]: b'{"item":{"user_id":1}}',
        Annotated[SimpleModel, "metadata"]: b'{"user_id":1}',
    }

    decoded = adapter.validate_json(model, payloads[model])

    if model == SimpleModel | None or model == Annotated[SimpleModel, "metadata"]:
        assert type(decoded) is SimpleModel
        assert adapter.is_model_instance(decoded, model) is True
    elif model == list[SimpleModel]:
        assert [type(item) for item in decoded] == [SimpleModel]
        assert adapter.is_model_instance(decoded, model) is True
    else:
        assert type(decoded["item"]) is SimpleModel
        assert adapter.is_model_instance(decoded, model) is True
