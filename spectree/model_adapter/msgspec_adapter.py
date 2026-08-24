import re
from dataclasses import is_dataclass
from types import UnionType
from typing import Annotated, Any, TypeAlias, Union, get_args, get_origin

import msgspec

from spectree.model_adapter.protocol import ModelAdapter, SchemaMode, ModelSpec
from spectree.models import ValidationErrorElement
from spectree.utils import get_model_key

_ERROR_PATH_RE = re.compile(r" - at `(?P<path>.+)`$")


MsgspecValidationError: TypeAlias = Annotated[
    list[ValidationErrorElement],
    msgspec.Meta(title="ValidationError"),
]

BaseFile = Annotated[
    Any,
    msgspec.Meta(
        extra_json_schema={
            "format": "binary",
            "type": "string",
        }
    ),
]


def _unwrap_annotated(model: ModelSpec) -> ModelSpec:
    while get_origin(model) is Annotated:
        model = get_args(model)[0]
    return model


def _is_instance_of_model(
    value: Any,
    model: ModelSpec,
) -> bool:
    model = _unwrap_annotated(model)
    origin = get_origin(model)

    checks = (
        (model is Any, True),
        (model is None or model is type(None), value is None),
        (
            origin in (Union, UnionType),
            any(_is_instance_of_model(value, option) for option in get_args(model)),
        ),
        (
            origin is list,
            (
                isinstance(value, list)
                and len(get_args(model)) == 1
                and all(
                    _is_instance_of_model(value_item, get_args(model)[0])
                    for value_item in value
                )
            ),
        ),
        (
            origin is tuple,
            _is_tuple_instance(value, get_args(model)),
        ),
        (
            origin is dict,
            _is_dict_instance(value, get_args(model)),
        ),
        (
            origin is set,
            (
                isinstance(value, set)
                and len(get_args(model)) == 1
                and all(
                    _is_instance_of_model(value_item, get_args(model)[0])
                    for value_item in value
                )
            ),
        ),
        (
            origin is frozenset,
            (
                isinstance(value, frozenset)
                and len(get_args(model)) == 1
                and all(
                    _is_instance_of_model(value_item, get_args(model)[0])
                    for value_item in value
                )
            ),
        ),
        (
            origin is not None and isinstance(origin, type),
            isinstance(value, origin),
        ),
        (
            isinstance(model, type),
            isinstance(value, model),
        ),
    )

    for matched, result in checks:
        if matched:
            return result

    return False


def _is_tuple_instance(
    value: Any,
    args: tuple[Any, ...],
) -> bool:
    if not isinstance(value, tuple):
        return False

    if len(args) == 2 and args[1] is Ellipsis:
        return all(_is_instance_of_model(item, args[0]) for item in value)

    return len(value) == len(args) and all(
        _is_instance_of_model(item, item_model)
        for item, item_model in zip(value, args, strict=True)
    )


def _is_dict_instance(
    value: Any,
    args: tuple[Any, ...],
) -> bool:
    return (
        isinstance(value, dict)
        and len(args) == 2
        and all(
            _is_instance_of_model(key, args[0]) and _is_instance_of_model(item, args[1])
            for key, item in value.items()
        )
    )


def _parse_error_location(message: str) -> list[str]:
    match = _ERROR_PATH_RE.search(message)
    if match is None:
        return []

    path = match.group("path")
    if path == "$":
        return []

    path = path.removeprefix("$")
    path = path.replace("[", ".").replace("]", "")
    return [part for part in path.split(".") if part]


class MsgspecCompiledModel:
    """Compiled msgspec runtime representation of a ModelSpec."""

    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec

    def is_instance(self, value: Any) -> bool:
        return _is_instance_of_model(value, self.model_spec)

    def validate_obj(self, value: Any) -> Any:
        return msgspec.convert(
            value,
            type=self.model_spec,
            strict=False,
        )

    def validate_json(self, value: bytes) -> Any:
        return msgspec.json.decode(
            value,
            type=self.model_spec,
            strict=False,
        )

    def dump_json(self, value: Any) -> bytes:
        return msgspec.json.encode(value)

    def json_schema(
        self,
        *,
        ref_template: str,
        mode: SchemaMode = "validation",
    ) -> dict[str, Any]:
        del mode

        model = (
            MsgspecValidationError
            if self.model_spec is msgspec.ValidationError
            else self.model_spec
        )

        ref_template = ref_template.replace(
            "{model}",
            "{name}",
        )

        schemas, components = msgspec.json.schema_components(
            (model,),
            ref_template=ref_template,
        )

        schema = schemas[0]

        ref = schema.get("$ref")
        if isinstance(ref, str):
            for key in tuple(components):
                if ref == ref_template.format(name=key):
                    schema = components.pop(key)
                    break

        if components:
            schema["$defs"] = components

        return schema


class MsgspecModelAdapter(
    ModelAdapter[
        Any,
        msgspec.ValidationError,
        BaseFile,
    ],
):
    """Msgspec model adapter."""

    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec

    def is_instance(self, value: Any) -> bool:
        model = self.model_spec
        origin = get_origin(model)
        self._compiled_models: dict[ModelSpec, MsgspecCompiledModel] = {}

    def compile(
        self,
        model: ModelSpec,
    ) -> MsgspecCompiledModel:
        try:
            compiled = self._compiled_models.get(model)
        except TypeError:
            return MsgspecCompiledModel(model)

        if compiled is None:
            compiled = MsgspecCompiledModel(model)
            self._compiled_models[model] = compiled

        return compiled

    def is_model_type(
        self,
        value: ModelSpec,
    ) -> bool:
        if value is msgspec.ValidationError:
            return True

        try:
            msgspec.inspect.type_info(value)
        except (
            TypeError,
            ValueError,
        ):
            return False

        return True

    def is_model_instance(
        self,
        value: Any,
        model: ModelSpec,
    ) -> bool:
        return self.compile(model).is_instance(value)

    def is_partial_model_instance(
        self,
        value: Any,
    ) -> bool:
        if not value:
            return False

        if isinstance(value, msgspec.Struct):
            return True

        if is_dataclass(value):
            return True

        if isinstance(value, dict):
            return any(
                self.is_partial_model_instance(key)
                or self.is_partial_model_instance(item)
                for key, item in value.items()
            )

        if isinstance(value, (list, tuple)):
            return any(self.is_partial_model_instance(item) for item in value)

        return False

    def validate_obj(
            self,
            model: ModelSpec,
            value: Any,
    ) -> Any:
        return self.compile(model).validate_obj(value)

    def validate_json(
            self,
            model: ModelSpec,
            value: bytes,
    ) -> Any:
        return self.compile(model).validate_json(value)

    def dump_json(
        self,
        value: Any,
    ) -> bytes:
        return self.encoder.encode(value)

    def make_root_model(
        self,
        root_type: ModelSpec,
        *,
        name: str | None = None,
        module: str | None = None,
    ) -> ModelSpec:
        del module

        model_name = name or "GeneratedRootModel"

        return Annotated[
            root_type,
            msgspec.Meta(title=model_name),
        ]

    def make_list_model(
        self,
        model: ModelSpec,
    ) -> ModelSpec:
        model_name = _model_name_for_generated_type(model)

        return self.make_root_model(
            list[model],  # type: ignore[valid-type]
            name=f"{model_name}List",
        )

    def json_schema(
            self,
            model: ModelSpec,
            *,
            ref_template: str,
            mode: SchemaMode = "validation",
    ) -> dict[str, Any]:
        return self.compile(model).json_schema(
            ref_template=ref_template,
            mode=mode,
        )

    def validation_errors(
        self,
        err: msgspec.ValidationError,
    ) -> list[dict[str, Any]]:
        message = str(err)

        return [
            {
                "loc": _parse_error_location(message),
                "msg": _ERROR_PATH_RE.sub("", message),
                "type": "validation_error",
            }
        ]


def _model_name_for_generated_type(
    model: ModelSpec,
) -> str:
    return get_model_key(model).split(".", 1)[0]
