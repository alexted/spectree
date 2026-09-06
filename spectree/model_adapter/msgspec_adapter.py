import re
from dataclasses import is_dataclass
from typing import Annotated, Any, TypeAlias

import msgspec

from spectree.model_adapter.protocol import (
    ModelAdapter,
    ModelSpec,
    SchemaMode,
)
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
        msgspec.convert(
            value,
            type=self.model_spec,
            strict=True,
        )
        return True

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
    ]
):
    """Msgspec model adapter."""

    validation_error = msgspec.ValidationError
    basefile = BaseFile

    def __init__(self) -> None:
        self.encoder = msgspec.json.Encoder()
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
        try:
            return self.compile(model).is_instance(value)
        except (
            msgspec.ValidationError,
            TypeError,
            ValueError,
        ):
            return False

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

    def dump_json(
        self,
        value: Any,
    ) -> bytes:
        return self.compile(type(value)).dump_json(value)

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


def _model_name_for_generated_type(model: ModelSpec) -> str:
    return get_model_key(model).split(".", 1)[0]
