from collections.abc import Mapping
from dataclasses import is_dataclass
from typing import Any, Sequence

from pydantic import BaseModel, RootModel, TypeAdapter, ValidationError
from pydantic_core import core_schema

from spectree.model_adapter.protocol import ModelAdapter, SchemaMode, ModelSpec
from spectree.models import ValidationErrorElement


class ValidationErrorType(RootModel[Sequence[ValidationErrorElement]]):
    """Model of a validation error response."""


class BaseFile:
    """
    An uploaded file, will be assigned as the corresponding web framework's
    file object.
    """

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: Mapping[str, Any], _handler):
        return {"format": "binary", "type": "string"}

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any, *_args, **_kwargs):
        return value


class PydanticCompiledModel:
    """Compiled Pydantic runtime representation of a ModelSpec."""

    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec

        self._is_base_model = (
            isinstance(model_spec, type)
            and issubclass(model_spec, BaseModel)
        )
        self._is_dataclass = (
            isinstance(model_spec, type)
            and is_dataclass(model_spec)
        )

        if self._is_base_model:
            self._type_adapter = None
        else:
            self._type_adapter = TypeAdapter(model_spec)

    def is_instance(self, value: Any) -> bool:
        if not (self._is_base_model or self._is_dataclass):
            return False

        return isinstance(value, self.model_spec)

    def validate_obj(self, value: Any) -> Any:
        if self._is_base_model:
            return self.model_spec.model_validate(value)

        return self._type_adapter.validate_python(value)

    def validate_json(self, value: bytes) -> Any:
        if self._is_base_model:
            return self.model_spec.model_validate_json(value)

        return self._type_adapter.validate_json(value)

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

    def dump_json(self, value: Any) -> bytes:
        if self._is_base_model:
            return self.model_spec.model_dump_json(value).encode("utf-8")

        return self._type_adapter.dump_json(value)

class PydanticModelAdapter(ModelAdapter[Any, ValidationError, type[BaseFile]]):
    """`pydantic` model adapter."""

    validation_error = ValidationError
    basefile = BaseFile

    def __init__(self) -> None:
        self._compiled_models: dict[ModelSpec, PydanticCompiledModel] = {}

    def compile(self, model: ModelSpec) -> PydanticCompiledModel:
        compiled = self._compiled_models.get(model)

        if compiled is None:
            compiled = PydanticCompiledModel(model)
            self._compiled_models[model] = compiled

        return compiled

    def is_model_type(self, value: ModelSpec) -> bool:
        if value is ValidationError:
            return True

        if not isinstance(value, type):
            try:
                TypeAdapter(value)
            except (TypeError, ValueError):
                return False
            return True

        return issubclass(value, BaseModel) or is_dataclass(value)

    def is_model_instance(
            self,
            value: Any,
            model: ModelSpec,
    ) -> bool:
        return self.compile(model).is_instance(value)

    def is_partial_model_instance(self, value: Any) -> bool:
        if not value:
            return False

        if isinstance(value, BaseModel) or is_dataclass(value):
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
        if not isinstance(value, BaseModel):
            value = self.validate_obj(type(value), value)

        return self.compile(type(value)).dump_json(value)

    def make_root_model(
        self,
        root_type: Any,
        *,
        name: str | None = None,
        module: str | None = None,
    ) -> ModelSpec:
        model_name = name or "GeneratedRootModel"
        module_name = module or __name__
        return type(model_name, (RootModel[root_type],), {"__module__": module_name})

    def make_list_model(self, model: ModelSpec) -> ModelSpec:
        return self.make_root_model(
            list[model],  # type: ignore[valid-type]
            name=f"{model.__name__}List",
            module=model.__module__,
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

    def validation_errors(self, err: ValidationError) -> Any:
        return err.errors(include_context=False)
