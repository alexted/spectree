import re
from dataclasses import is_dataclass
from typing import Annotated, Any, TypeAlias, get_args, get_origin

import msgspec

from spectree.model_adapter.protocol import ModelAdapter, SchemaMode, ModelSpec
from spectree.models import ValidationErrorElement

_ERROR_PATH_RE = re.compile(r" - at `(?P<path>.+)`$")

MsgspecValidationError: TypeAlias = Annotated[
    list[ValidationErrorElement], msgspec.Meta(title="ValidationError")
]


BaseFile = Annotated[
    Any, msgspec.Meta(extra_json_schema={"format": "binary", "type": "string"})
]


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
        model = self.model_spec
        origin = get_origin(model)

        while origin is Annotated:
            model = get_args(model)[0]
            origin = get_origin(model)

        if origin is list:
            item_model = get_args(model)[0]

            return (
                isinstance(value, list)
                and isinstance(item_model, type)
                and all(isinstance(item, item_model) for item in value)
            )

        if not isinstance(model, type):
            return False

        if issubclass(model, msgspec.Struct):
            return isinstance(value, model)

        if is_dataclass(model):
            return isinstance(value, model)

        return False

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

    def json_schema(
        self,
        *,
        ref_template: str,
        mode: SchemaMode = "validation",
    ) -> dict[str, Any]:
        if self.model_spec is msgspec.ValidationError:
            model = MsgspecValidationError
        else:
            model = self.model_spec

        ref_template = ref_template.replace("{model}", "{name}")

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


class MsgspecModelAdapter(ModelAdapter[Any, msgspec.ValidationError, BaseFile]):
    """`msgspec` model adapter."""

    validation_error = msgspec.ValidationError
    basefile = BaseFile

    def __init__(self) -> None:
        self.encoder = msgspec.json.Encoder()

    def is_model_type(self, value: ModelSpec) -> bool:
        """All kinds of types are treated the same."""
        return True

    def is_model_instance(
            self,
            value: Any,
            model: ModelSpec,
    ) -> bool:
        return self.compile(model).is_instance(value)

    def is_partial_model_instance(self, value: Any) -> bool:
        if not value:
            return False

        if isinstance(value, msgspec.Struct) or is_dataclass(value):
            return True

        if isinstance(value, dict):
            return any(
                self.is_partial_model_instance(key)
                or self.is_partial_model_instance(item)
                for key, item in value.items()
            )

        if isinstance(value, (list, tuple)):
            return any(
                self.is_partial_model_instance(item)
                for item in value
            )

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

    def dump_json(self, value: Any) -> bytes:
        return self.encoder.encode(value)

    def make_root_model(
        self,
        root_type: ModelSpec,
        *,
        name: str | None = None,
        module: str | None = None,
    ) -> ModelSpec:
        """
        All the types are treated the same in `msgspec`.

        See: https://github.com/jcrist/msgspec/issues/484
        """
        model_name = name or "GeneratedRootModel"
        T = Annotated[root_type, msgspec.Meta(title=model_name)]  # type: ignore
        return T  # type: ignore

    def make_list_model(self, model: ModelSpec) -> ModelSpec:
        list_model = self.make_root_model(list[model], name=f"{model.__name__}List")  # type: ignore
        return list_model

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

    def validation_errors(self, err: msgspec.ValidationError):
        """Expect a `list[ValidationErrorElement]`"""
        message = str(err)
        return [
            {
                "loc": _parse_error_location(message),
                "msg": _ERROR_PATH_RE.sub("", message),
                "type": "validation_error",
            }
        ]

    def compile(self, model: ModelSpec) -> MsgspecCompiledModel:
        return MsgspecCompiledModel(model)
