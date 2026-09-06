from dataclasses import dataclass
from typing import Any, Iterator, Mapping, TypeAlias

from spectree.model_adapter.protocol import SchemaMode
from spectree.utils import json_compatible_deepcopy

SchemaName: TypeAlias = str


class SchemaCollisionError(ValueError):
    """Raised when incompatible schemas resolve to the same component name."""


@dataclass(frozen=True, slots=True)
class SchemaRecord:
    """Immutable description of a registered top-level schema."""

    model: Any
    mode: SchemaMode
    component_name: SchemaName
    raw_schema: dict[str, Any]
    schema: dict[str, Any]


class SchemaRegistry(Mapping[str, dict[str, Any]]):
    """
    Registry of OpenAPI component schemas.

    Registration is keyed by:

        (naming_strategy(model), schema_mode)

    Validation and serialization schemas for the same model share a component
    when their raw schemas are identical.

    When they differ, validation uses <name> and serialization uses
    <name>.serialization.

    Component iteration and snapshots are sorted by component name, so their
    observable order is independent of registration order.
    """

    def __init__(
        self,
        naming_strategy,
        nested_naming_strategy,
    ) -> None:
        self.naming_strategy = naming_strategy
        self.nested_naming_strategy = nested_naming_strategy

        self._schemas: dict[SchemaName, dict[str, Any]] = {}
        self._records: dict[
            tuple[SchemaName, SchemaMode],
            SchemaRecord,
        ] = {}

    def __getitem__(
        self,
        name: str,
    ) -> dict[str, Any]:
        """
        Return a detached copy of the registered schema.
        """
        return json_compatible_deepcopy(
            self._schemas[name],
        )

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over component names in deterministic order.
        """
        return iter(sorted(self._schemas))

    def __len__(self) -> int:
        return len(self._schemas)

    def register(
        self,
        model: Any,
        mode: SchemaMode,
        schema: dict[str, Any],
    ) -> str:
        """
        Register a model schema and return its OpenAPI component name.

        Registration is idempotent for the same model, mode and schema.

        Validation and serialization schemas for the same model share a
        component when their raw schemas are identical.

        When the schemas differ, the names are deterministic:

            validation     -> <name>
            serialization  -> <name>.serialization
        """
        self._validate_mode(mode)

        base_name = self.naming_strategy(model)
        if not isinstance(base_name, str) or not base_name:
            raise ValueError("naming_strategy must return a non-empty string")

        raw_schema = json_compatible_deepcopy(schema)
        identity = (base_name, mode)

        existing = self._records.get(identity)
        if existing is not None:
            if existing.model is not model:
                raise SchemaCollisionError(
                    f"Schema name collision for {base_name!r}: "
                    f"multiple models use the same name for {mode!r} schema."
                )

            if existing.raw_schema != raw_schema:
                raise SchemaCollisionError(
                    f"Schema for {base_name!r} in {mode!r} mode "
                    "changed after registration."
                )

            return existing.component_name

        other_mode = self._other_mode(mode)
        other = self._records.get((base_name, other_mode))

        if other is not None:
            if other.model is not model:
                raise SchemaCollisionError(
                    f"Schema name collision for {base_name!r}: "
                    "different models use the same component name."
                )

            if other.raw_schema == raw_schema:
                record = SchemaRecord(
                    model=model,
                    mode=mode,
                    component_name=other.component_name,
                    raw_schema=raw_schema,
                    schema=json_compatible_deepcopy(other.schema),
                )
                self._records[identity] = record
                return other.component_name

        component_name = self._component_name(
            base_name=base_name,
            mode=mode,
        )

        self._ensure_component_available(
            component_name=component_name,
            model=model,
            mode=mode,
        )

        normalized = self._normalize_schema(
            raw_schema,
            component_name,
        )

        self._schemas[component_name] = normalized
        self._records[identity] = SchemaRecord(
            model=model,
            mode=mode,
            component_name=component_name,
            raw_schema=raw_schema,
            schema=json_compatible_deepcopy(normalized),
        )

        return component_name

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """
        Return a detached deterministic snapshot of all registered schemas.
        """
        return {
            name: json_compatible_deepcopy(self._schemas[name])
            for name in sorted(self._schemas)
        }

    def _ensure_component_available(
        self,
        *,
        component_name: str,
        model: Any,
        mode: SchemaMode,
    ) -> None:
        if component_name not in self._schemas:
            return

        for record in self._records.values():
            if record.component_name != component_name:
                continue

            if record.model is model and record.mode == mode:
                return

            raise SchemaCollisionError(
                f"Schema component {component_name!r} already exists "
                "for a different model or schema mode."
            )

        raise SchemaCollisionError(
            f"Schema component {component_name!r} already exists."
        )

    @staticmethod
    def _component_name(
        *,
        base_name: str,
        mode: SchemaMode,
    ) -> str:
        if mode == "validation":
            return base_name

        return f"{base_name}.serialization"

    @staticmethod
    def _validate_mode(
        mode: SchemaMode,
    ) -> None:
        if mode not in ("validation", "serialization"):
            raise ValueError(f"Unsupported schema mode: {mode!r}")

    @staticmethod
    def _other_mode(
        mode: SchemaMode,
    ) -> SchemaMode:
        if mode == "validation":
            return "serialization"

        return "validation"

    def _normalize_schema(
        self,
        schema: dict[str, Any],
        component_name: str,
    ) -> dict[str, Any]:
        """
        Normalize adapter-local schema references.

        Local $defs references are converted to OpenAPI component references.
        Nested definitions are processed recursively.
        """
        self._normalize_schema_node(
            schema,
            component_name,
        )
        return schema

    def _normalize_schema_node(
        self,
        node: dict[str, Any],
        component_name: str,
    ) -> None:
        definitions = node.get("$defs")
        if not isinstance(definitions, dict):
            return

        replacements: dict[str, str] = {}

        for key in definitions:
            nested_name = self._nested_component_name(
                component_name,
                key,
            )

            replacements[f"#/components/schemas/{key}"] = (
                f"#/components/schemas/{nested_name}"
            )

        self._replace_refs(
            node,
            replacements,
        )

        for key, definition in definitions.items():
            if not isinstance(definition, dict):
                continue

            child_component_name = self._nested_component_name(
                component_name,
                key,
            )

            self._normalize_schema_node(
                definition,
                child_component_name,
            )

    def _nested_component_name(
        self,
        parent_name: str,
        child_name: str,
    ) -> str:
        result = self.nested_naming_strategy(
            parent_name,
            child_name,
        )

        if not isinstance(result, str) or not result:
            raise ValueError("nested_naming_strategy must return a non-empty string")

        return result

    @classmethod
    def _replace_refs(
        cls,
        value: Any,
        replacements: Mapping[str, str],
    ) -> None:
        if isinstance(value, dict):
            ref = value.get("$ref")

            if isinstance(ref, str):
                replacement = replacements.get(ref)
                if replacement is not None:
                    value["$ref"] = replacement

            for nested in value.values():
                cls._replace_refs(
                    nested,
                    replacements,
                )

        elif isinstance(value, list):
            for nested in value:
                cls._replace_refs(
                    nested,
                    replacements,
                )
