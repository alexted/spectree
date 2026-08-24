from dataclasses import dataclass
from typing import Annotated, List

import pytest

from tests.common_pydantic import DemoModel

pytest.importorskip("pydantic")

from pydantic import BaseModel

from spectree.model_adapter import get_pydantic_model_adapter

ADAPTER = get_pydantic_model_adapter()

DummyRootModel = ADAPTER.make_root_model(List[int], name="DummyRootModel")

NestedRootModel = ADAPTER.make_root_model(DummyRootModel, name="NestedRootModel")


class SimpleModel(BaseModel):
    user_id: int


@dataclass
class RootModelLookalike:
    __root__: List[str]


@pytest.mark.parametrize(
    "value, expected",
    [
        (DummyRootModel, False),
        (DummyRootModel.model_validate([1, 2, 3]), True),
        (NestedRootModel, False),
        (
            NestedRootModel.model_validate(DummyRootModel.model_validate([1, 2, 3])),
            True,
        ),
        (SimpleModel, False),
        (SimpleModel(user_id=1), True),
        (RootModelLookalike, False),
        (RootModelLookalike(__root__=["False"]), False),
        (list, False),
        ([1, 2, 3], False),
        (str, False),
        ("str", False),
        (int, False),
        (1, False),
    ],
)
def test_is_base_model_instance(value, expected):
    assert ADAPTER.is_model_instance(value, BaseModel) is expected


@dataclass
class DemoDataclass:
    value: int = 0


def test_pydantic_model_spec_types(
    model_adapter,
):
    assert model_adapter.is_model_type(DemoModel)
    assert model_adapter.is_model_type(DemoDataclass)


def test_pydantic_model_spec_generic_aliases(model_adapter):
    assert model_adapter.is_model_type(list[int])
    assert model_adapter.is_model_type(dict[str, int])
    assert model_adapter.is_model_type(int | None)


def test_pydantic_rejects_unsupported_model_spec(model_adapter):
    class Unsupported:
        value: int

    assert model_adapter.is_model_type(Unsupported) is False


def test_pydantic_accepts_nested_generic_model_spec(model_adapter):
    spec = list[dict[str, SimpleModel]]

    assert model_adapter.is_model_type(spec)


def test_pydantic_is_model_instance_generic_aliases(model_adapter):
    model = SimpleModel

    users = [
        SimpleModel(user_id=1),
        SimpleModel(user_id=2),
    ]

    assert model_adapter.is_model_instance(
        users,
        list[model],
    )

    assert not model_adapter.is_model_instance(
        [
            {"user_id": 1},
        ],
        list[model],
    )


def test_pydantic_annotated_model_spec(model_adapter):
    spec = Annotated[
        SimpleModel,
        "metadata",
    ]

    assert model_adapter.is_model_type(spec)

    value = model_adapter.validate_obj(
        spec,
        {"user_id": "1"},
    )

    assert isinstance(value, SimpleModel)


def test_pydantic_generic_model_json_roundtrip(model_adapter):
    spec = list[SimpleModel]

    value = model_adapter.validate_obj(
        spec,
        [
            {"user_id": 1},
            {"user_id": 2},
        ],
    )

    payload = model_adapter.dump_json(value)

    restored = model_adapter.validate_json(
        spec,
        payload,
    )

    assert model_adapter.is_model_instance(
        restored,
        spec,
    )


def test_pydantic_rejects_unsupported_nested_generic(model_adapter):
    class Unsupported:
        value: int

    assert model_adapter.is_model_type(list[Unsupported]) is False


@pytest.mark.pydantic
def test_pydantic_compilation_is_cached():
    adapter = get_pydantic_model_adapter()

    model = adapter.make_root_model(
        list[int],
        name="Numbers",
    )

    first = adapter.compile(model)
    second = adapter.compile(model)

    assert first is second


@pytest.mark.pydantic
def test_pydantic_compiled_generic_model_is_not_direct_instance():
    adapter = get_pydantic_model_adapter()

    model = list[int]
    compiled = adapter.compile(model)

    assert compiled.is_instance([1, 2, 3]) is False


@pytest.mark.pydantic
def test_pydantic_compiled_generic_model_validates():
    adapter = get_pydantic_model_adapter()

    model = list[int]
    compiled = adapter.compile(model)

    assert compiled.validate_obj(["1", "2"]) == [1, 2]
