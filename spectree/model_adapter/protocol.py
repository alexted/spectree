from typing import Any, Literal, Protocol, TypeAlias, TypeVar

ModelClass: TypeAlias = type[Any]

# A ModelSpec is an adapter-supported Python type expression.
#
# Examples include:
#
#     User
#     list[User]
#     dict[str, User]
#     Annotated[User, ...]
#     User | None
#
# The concrete set of supported expressions is defined by the selected
# ModelAdapter.
ModelSpec: TypeAlias = Any

ModelT = TypeVar("ModelT")
ValidationErrorT = TypeVar("ValidationErrorT", bound=Exception)
BaseFileT = TypeVar("BaseFileT")

SchemaMode: TypeAlias = Literal["validation", "serialization"]


class ModelAdapter(Protocol[ModelT, ValidationErrorT, BaseFileT]):
    """Adapter contract for validation, serialization and schema generation."""

    validation_error: type[ValidationErrorT]
    basefile: BaseFileT

    def is_model_type(self, value: ModelSpec) -> bool:
        """Return whether ``value`` is a supported model/type expression."""
        ...

    def is_model_instance(
        self,
        value: Any,
        model: ModelSpec,
    ) -> bool:
        """
        Return whether ``value`` is already a valid instance of ``model``.

        Returning ``True`` allows runtime validation/serialization paths to
        avoid reconstructing an already valid model instance.
        """
        ...

    def is_partial_model_instance(self, value: Any) -> bool: ...

    def validate_obj(
        self,
        model: ModelSpec,
        value: Any,
    ) -> ModelT: ...

    def validate_json(
        self,
        model: ModelSpec,
        value: bytes,
    ) -> ModelT: ...

    def dump_json(self, value: Any) -> bytes: ...

    def make_root_model(
        self,
        root_type: ModelSpec,
        *,
        name: str | None = None,
        module: str | None = None,
    ) -> ModelSpec: ...

    def make_list_model(
        self,
        model: ModelSpec,
    ) -> ModelSpec: ...

    def json_schema(
        self,
        model: ModelSpec,
        *,
        ref_template: str,
        mode: SchemaMode = "validation",
    ) -> dict[str, Any]: ...

    def validation_errors(
        self,
        err: ValidationErrorT,
    ) -> Any: ...
