from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from types import GenericAlias
from typing import (
    Annotated,
    Any,
    Callable,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pytest

from spectree._types import ModelAdapterType
from spectree.model_adapter import (
    get_msgspec_model_adapter,
    get_pydantic_model_adapter,
)

PYDANTIC_MODEL_CASE_PARAMS = [
    pytest.param("pydantic", marks=pytest.mark.pydantic, id="pydantic"),
]
MSGSPEC_MODEL_CASE_PARAMS = [
    pytest.param("msgspec", marks=pytest.mark.msgspec, id="msgspec"),
]
MODEL_CASE_PARAMS = [
    *PYDANTIC_MODEL_CASE_PARAMS,
    *MSGSPEC_MODEL_CASE_PARAMS,
]

MODEL_DEFINITION_CACHE_SIZE = 128
ModelResolver = Callable[[Any | None, str | None], Any | None]


@dataclass
class SimpleModel:
    user_id: int


@dataclass
class RootModelLookalike:
    __root__: list[str]


@dataclass(frozen=True)
class ModelCase:
    name: str
    adapter: ModelAdapterType
    _get_model: ModelResolver
    root_model_lookalike: type[RootModelLookalike] = RootModelLookalike

    def get_model(
        self,
        model_def: Any | None,
        *,
        name: str | None = None,
    ) -> Any | None:
        return self._get_model(model_def, name)

    def validate_obj(self, model: Any, value: Any) -> Any:
        return self.adapter.validate_obj(model, value)

    def validate_json(self, model: Any, value: bytes) -> Any:
        return self.adapter.validate_json(model, value)

    def dump_python(self, value: Any) -> Any:
        return json.loads(self.adapter.dump_json(value))

    def list_of(self, model: Any) -> Any:
        return GenericAlias(list, (model,))


def _dataclass_field_types(model_def: type[Any]) -> list[tuple[Any, Any]]:
    if not is_dataclass(model_def):
        raise TypeError(f"{model_def!r} is not a dataclass")

    type_hints = get_type_hints(model_def, include_extras=True)
    return [
        (model_field, type_hints.get(model_field.name, model_field.type))
        for model_field in fields(model_def)
    ]


def _build_model_resolver(
    adapter: ModelAdapterType,
) -> ModelResolver:
    def convert_type_def(type_def: Any) -> Any:
        origin = get_origin(type_def)

        if origin is None:
            return type_def

        args = tuple(convert_type_def(arg) for arg in get_args(type_def))

        if origin is Union:
            return Union[args]

        if origin is Annotated:
            return Annotated[args]

        return GenericAlias(origin, args)

    @lru_cache(maxsize=MODEL_DEFINITION_CACHE_SIZE)
    def get_model(
        model_def: Any | None,
        name: str | None,
    ) -> Any | None:
        if model_def is None:
            return None

        converted_def = convert_type_def(model_def)
        if is_dataclass(model_def):
            return converted_def

        origin = get_origin(model_def)
        if origin is list and name is None:
            item_model = get_args(converted_def)[0]
            return adapter.make_list_model(item_model)

        return adapter.make_root_model(converted_def, name=name, module=__name__)

    return cast(ModelResolver, get_model)


def build_model_case(name: str) -> ModelCase:
    if name == "pydantic":
        return _build_pydantic_case()
    if name == "msgspec":
        return _build_msgspec_case()
    raise ValueError(f"unknown model adapter case: {name}")


def _build_pydantic_case() -> ModelCase:
    if importlib.util.find_spec("pydantic") is None:
        pytest.skip("pydantic is not installed")

    adapter = get_pydantic_model_adapter()

    return ModelCase(
        name="pydantic",
        adapter=adapter,
        _get_model=_build_model_resolver(adapter),
    )


def _build_msgspec_case() -> ModelCase:
    if importlib.util.find_spec("msgspec") is None:
        pytest.skip("msgspec is not installed")

    adapter = get_msgspec_model_adapter()

    return ModelCase(
        name="msgspec",
        adapter=adapter,
        _get_model=_build_model_resolver(adapter),
    )
