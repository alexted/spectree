from typing import Any, Literal, Protocol, TypeAlias, TypeVar

ModelClass: TypeAlias = type[Any]
ModelSpec: TypeAlias = Any

ModelT = TypeVar("ModelT")
ValidationErrorT = TypeVar("ValidationErrorT", bound=Exception)
BaseFileT = TypeVar("BaseFileT")

SchemaMode: TypeAlias = Literal["validation", "serialization"]


class CompiledModel(Protocol[ModelT]):
    """Adapter-specific runtime representation of a ModelSpec."""

    model_spec: ModelSpec

    def is_instance(self, value: Any) -> bool:
        """Return whether value is already a valid instance of this model."""
        ...

    def validate_obj(self, value: Any) -> ModelT:
        """Validate an already decoded Python value."""
        ...

    def validate_json(self, value: bytes) -> ModelT:
        """Validate a JSON payload."""
        ...

    def dump_json(self, value: Any) -> bytes:
        """Serialize a value to JSON."""
        ...

    def json_schema(
        self,
        *,
        ref_template: str,
        mode: SchemaMode = "validation",
    ) -> dict[str, Any]:
        """Generate the JSON schema for the compiled model."""
        ...


class ModelAdapter(Protocol[ModelT, ValidationErrorT, BaseFileT]):
    """The protocol of model adapter."""

    validation_error: type[ValidationErrorT]
    basefile: BaseFileT

    def is_model_type(self, value: ModelSpec) -> bool:
        """Return whether value is a supported model/type expression."""
        ...

    def is_model_instance(
        self,
        value: Any,
        model: ModelSpec,
    ) -> bool:
        """Check if value is an instance of model under this adapter."""
        ...

    def is_partial_model_instance(self, value: Any) -> bool:
        """Return whether value contains a model instance."""
        ...

    def validate_obj(
        self,
        model: ModelSpec,
        value: Any,
    ) -> ModelT:
        """Validate a Python object against a model specification."""
        ...

    def validate_json(
        self,
        model: ModelSpec,
        value: bytes,
    ) -> ModelT:
        """Validate JSON bytes against a model specification."""
        ...

    def dump_json(self, value: Any) -> bytes:
        """Serialize a value to JSON."""
        ...

    def make_root_model(
        self,
        root_type: ModelSpec,
        *,
        name: str | None = None,
        module: str | None = None,
    ) -> ModelSpec:
        """Create an adapter-specific root model."""
        ...

    def make_list_model(
        self,
        model: ModelSpec,
    ) -> ModelSpec:
        """Create an adapter-specific list model."""
        ...

    def json_schema(
        self,
        model: ModelSpec,
        *,
        ref_template: str,
        mode: SchemaMode = "validation",
    ) -> dict[str, Any]:
        """Generate the JSON schema for a model specification."""
        ...

    def validation_errors(
        self,
        err: ValidationErrorT,
    ) -> Any:
        """Convert an adapter validation error to Spectree's error format."""
        ...

    def compile(
        self,
        model: ModelSpec,
    ) -> CompiledModel[ModelT]:
        """Compile a model specification into an adapter-specific runtime model."""
        ...
