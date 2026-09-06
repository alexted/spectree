from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from functools import cache
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    PydanticUserError,
    RootModel,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import core_schema

from spectree.model_adapter.protocol import (
    ModelAdapter,
    ModelSpec,
    SchemaMode,
)
from spectree.models import ValidationErrorElement
from spectree.utils import get_model_key


class ValidationErrorType(RootModel[Sequence[ValidationErrorElement]]):
    """Model of a validation error response."""


class BaseFile:
    """
    An uploaded file, will be assigned as the corresponding web framework's
    file object.
    """

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: Mapping[str, Any],
        _handler,
    ) -> dict[str, str]:
        return {
            "format": "binary",
            "type": "string",
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type,
        _handler,
    ):
        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(
        cls,
        value: Any,
        *_args,
        **_kwargs,
    ) -> Any:
        return value


def _unwrap_annotated(model: ModelSpec) -> ModelSpec:
    while get_origin(model) is Annotated:
        model = get_args(model)[0]
    return model


def _is_base_model_type(model: ModelSpec) -> bool:
    return isinstance(model, type) and issubclass(model, BaseModel)


def _is_instance_of_model(
    value: Any,
    model: ModelSpec,
) -> bool:
    model = _unwrap_annotated(model)
    origin = get_origin(model)

    if model is Any:
        return True

    if model is None or model is type(None):
        return value is None

    if origin in (Union, UnionType):
        return any(
            _is_instance_of_model(value, option)
            for option in get_args(model)
        )

    if origin is Literal:
        return any(value == literal for literal in get_args(model))

    if origin is list:
        args = get_args(model)
        return (
            isinstance(value, list)
            and len(args) == 1
            and all(
                _is_instance_of_model(item, args[0])
                for item in value
            )
        )

    if origin is tuple:
        args = get_args(model)

        if not isinstance(value, tuple):
            return False

        if len(args) == 2 and args[1] is Ellipsis:
            return all(
                _is_instance_of_model(item, args[0])
                for item in value
            )

        return (
            len(value) == len(args)
            and all(
                _is_instance_of_model(item, item_model)
                for item, item_model in zip(value, args, strict=True)
            )
        )

    if origin is dict:
        args = get_args(model)

        return (
            isinstance(value, dict)
            and len(args) == 2
            and all(
                _is_instance_of_model(key, args[0])
                and _is_instance_of_model(item, args[1])
                for key, item in value.items()
            )
        )

    if origin is set:
        args = get_args(model)

        return (
            isinstance(value, set)
            and len(args) == 1
            and all(
                _is_instance_of_model(item, args[0])
                for item in value
            )
        )

    if origin is frozenset:
        args = get_args(model)

        return (
            isinstance(value, frozenset)
            and len(args) == 1
            and all(
                _is_instance_of_model(item, args[0])
                for item in value
            )
        )

    if origin is not None:
        return isinstance(value, origin)

    if isinstance(model, type):
        return isinstance(value, model)

    return False


class PydanticCompiledModel:
    """Compiled Pydantic runtime representation of a ModelSpec."""

    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec
        self._is_base_model = _is_base_model_type(model_spec)
        self._is_dataclass = (
            isinstance(model_spec, type)
            and is_dataclass(model_spec)
        )

        if self._is_base_model or model_spec is ValidationError:
            self._type_adapter = None
        else:
            self._type_adapter = PydanticModelAdapter._type_adapter(model_spec)

    def is_instance(self, value: Any) -> bool:
        if self.model_spec is ValidationError:
            return isinstance(value, ValidationError)

        if self._is_base_model or self._is_dataclass:
            return isinstance(value, self.model_spec)

        return _is_instance_of_model(value, self.model_spec)

    def validate_obj(self, value: Any) -> Any:
        if self._is_base_model:
            return self.model_spec.model_validate(value)

        if self.model_spec is ValidationError:
            return value

        return self._type_adapter.validate_python(value)

    def validate_json(self, value: bytes) -> Any:
        if self._is_base_model:
            return self.model_spec.model_validate_json(value)

        if self.model_spec is ValidationError:
            return ValidationErrorType.model_validate_json(value)

        return self._type_adapter.validate_json(value)

    def dump_json(self, value: Any) -> bytes:
        if isinstance(value, BaseModel):
            return value.model_dump_json().encode("utf-8")

        return PydanticModelAdapter._type_adapter(
            type(value),
        ).dump_json(value)

    def json_schema(
        self,
        *,
        ref_template: str,
        mode: SchemaMode = "validation",
    ) -> dict[str, Any]:
        if self._is_base_model:
            return self.model_spec.model_json_schema(
                ref_template=ref_template,
                mode=mode,
            )

        if self.model_spec is ValidationError:
            return ValidationErrorType.model_json_schema(
                ref_template=ref_template,
                mode=mode,
            )

        return self._type_adapter.json_schema(
            ref_template=ref_template,
            mode=mode,
        )


class PydanticModelAdapter(
    ModelAdapter[Any, ValidationError, type[BaseFile]],
):
    """Pydantic model adapter."""

    validation_error = ValidationError
    basefile = BaseFile

    def __init__(self) -> None:
        self._compiled_models: dict[ModelSpec, PydanticCompiledModel] = {}

    def compile(
        self,
        model: ModelSpec,
    ) -> PydanticCompiledModel:
        try:
            compiled = self._compiled_models.get(model)
        except TypeError:
            return PydanticCompiledModel(model)

        if compiled is None:
            compiled = PydanticCompiledModel(model)
            self._compiled_models[model] = compiled

        return compiled

    @staticmethod
    @cache
    def _cached_type_adapter(
        model: ModelSpec,
    ) -> TypeAdapter[Any]:
        return TypeAdapter(model)

    @staticmethod
    def _type_adapter(
        model: ModelSpec,
    ) -> TypeAdapter[Any]:
        try:
            hash(model)
        except TypeError:
            return TypeAdapter(model)

        return PydanticModelAdapter._cached_type_adapter(model)

    def is_model_type(
        self,
        value: ModelSpec,
    ) -> bool:
        if value is ValidationError:
            return True

        try:
            self._type_adapter(value)
        except (
            PydanticUserError,
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

        if isinstance(value, BaseModel):
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
        model_name = name or "GeneratedRootModel"
        module_name = module or __name__

        return type(
            model_name,
            (RootModel[root_type],),
            {"__module__": module_name},
        )

    def make_list_model(
        self,
        model: ModelSpec,
    ) -> ModelSpec:
        model_name = _model_name_for_generated_type(model)

        return self.make_root_model(
            list[model],  # type: ignore[valid-type]
            name=f"{model_name}List",
            module=getattr(model, "__module__", __name__),
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
        err: ValidationError,
    ) -> list[dict[str, Any]]:
        return err.errors(include_context=False)


def _model_name_for_generated_type(
    model: ModelSpec,
) -> str:
    return get_model_key(model).split(".", 1)[0]
