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

    Registration is keyed by the pair:

        (naming_strategy(model), schema_mode)

    Validation and serialization schemas for the same model share a component
    when their raw schemas are identical.

    When they differ, component names are deterministic:

        validation     -> <name>
        serialization  -> <name>.serialization

    Registration order must not affect the resulting component names.
    """

    def __init__(
        self,
        naming_strategy,
        nested_naming_strategy,
    ) -> None:
        self.naming_strategy = naming_strategy
        self.nested_naming_strategy = nested_naming_strategy

        self._schemas: dict[SchemaName, dict[str, Any]] = {}
        self._records: dict[tuple[SchemaName, SchemaMode], SchemaRecord] = {}

    def __getitem__(self, name: str) -> dict[str, Any]:
        """
        Return a defensive copy of the registered schema.

        The registry owns its internal schema state and callers must not be
        able to mutate it accidentally.
        """
        return json_compatible_deepcopy(self._schemas[name])

    def __iter__(self) -> Iterator[str]:
        return iter(self._schemas)

    def __len__(self) -> int:
        return len(self._schemas)

    def register(
        self,
        model: Any,
        mode: SchemaMode,
        schema: dict[str, Any],
    ) -> str:
        """
        Register a model schema and return its final OpenAPI component name.

        Registration is idempotent for the same model/mode/schema tuple.

        For different validation/serialization schemas of the same model the
        final names are deterministic regardless of registration order.
        """
        self._validate_mode(mode)

        base_name = self.naming_strategy(model)
        if not isinstance(base_name, str) or not base_name:
            raise ValueError(
                "naming_strategy must return a non-empty string"
            )

        raw_schema = json_compatible_deepcopy(schema)
        identity = (base_name, mode)
        existing = self._records.get(identity)

        if existing is not None:
            normalized = self._normalize_schema(
                raw_schema,
                existing.component_name,
            )

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

            if existing.schema != normalized:
                raise SchemaCollisionError(
                    f"Normalized schema for {base_name!r} in {mode!r} mode "
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
                    schema=other.schema,
                )
                self._records[identity] = record
                return other.component_name

            return self._register_distinct_mode(
                model=model,
                mode=mode,
                base_name=base_name,
                raw_schema=raw_schema,
                other=other,
            )

        component_name = self._default_component_name(
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
            schema=normalized,
        )

        return component_name

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """
        Return a detached snapshot of all registered top-level schemas.
        """
        return json_compatible_deepcopy(self._schemas)

    def _register_distinct_mode(
        self,
        *,
        model: Any,
        mode: SchemaMode,
        base_name: str,
        raw_schema: dict[str, Any],
        other: SchemaRecord,
    ) -> str:
        """
        Register two different schemas for the same model.

        The canonical naming policy is:

            validation     -> base_name
            serialization  -> base_name + ".serialization"

        If serialization was registered first and currently occupies the base
        component, move it to its deterministic final component before adding
        validation.
        """
        validation_name = base_name
        serialization_name = f"{base_name}.serialization"

        desired_name = (
            validation_name
            if mode == "validation"
            else serialization_name
        )

        if (
            mode == "validation"
            and other.mode == "serialization"
            and other.component_name == validation_name
        ):
            self._rename_component(
                record=other,
                new_name=serialization_name,
            )

        self._ensure_component_available(
            component_name=desired_name,
            model=model,
            mode=mode,
        )

        normalized = self._normalize_schema(
            raw_schema,
            desired_name,
        )

        self._schemas[desired_name] = normalized
        self._records[(base_name, mode)] = SchemaRecord(
            model=model,
            mode=mode,
            component_name=desired_name,
            raw_schema=raw_schema,
            schema=normalized,
        )

        return desired_name

    def _rename_component(
        self,
        *,
        record: SchemaRecord,
        new_name: str,
    ) -> None:
        """
        Move a previously registered component to its deterministic name.

        All registered schema references to the old component are updated.
        The moved schema is re-normalized so nested component names follow the
        new parent component name.
        """
        old_name = record.component_name

        if old_name == new_name:
            return

        self._ensure_component_available(
            component_name=new_name,
            model=record.model,
            mode=record.mode,
        )

        old_ref = f"#/components/schemas/{old_name}"
        new_ref = f"#/components/schemas/{new_name}"

        moved_schema = self._normalize_schema(
            json_compatible_deepcopy(record.raw_schema),
            new_name,
        )

        del self._schemas[old_name]
        self._schemas[new_name] = moved_schema

        for identity, current in tuple(self._records.items()):
            if (
                    current.model is record.model
                    and current.mode == record.mode
                    and current.component_name == old_name
            ):
                self._records[identity] = SchemaRecord(
                    model=current.model,
                    mode=current.mode,
                    component_name=new_name,
                    raw_schema=json_compatible_deepcopy(
                        current.raw_schema
                    ),
                    schema=json_compatible_deepcopy(
                        moved_schema
                    ),
                )
                break

        self._rewrite_registered_refs(
            old_ref,
            new_ref,
        )

    def _rewrite_registered_refs(
        self,
        old_ref: str,
        new_ref: str,
    ) -> None:
        """
        Rewrite references to a renamed top-level component in all schemas.
        """
        for name, schema in tuple(self._schemas.items()):
            self._replace_ref_prefix(
                schema,
                old_ref=old_ref,
                new_ref=new_ref,
            )

            for identity, record in tuple(self._records.items()):
                if record.component_name != name:
                    continue
                self._records[identity] = SchemaRecord(
                    model=record.model,
                    mode=record.mode,
                    component_name=record.component_name,
                    raw_schema=record.raw_schema,
                    schema=json_compatible_deepcopy(schema),
                )

    @classmethod
    def _replace_ref_prefix(
        cls,
        value: Any,
        *,
        old_ref: str,
        new_ref: str,
    ) -> None:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str):
                if ref == old_ref:
                    value["$ref"] = new_ref
                elif ref.startswith(old_ref + "."):
                    value["$ref"] = (
                        new_ref + ref[len(old_ref):]
                    )

            for nested in value.values():
                cls._replace_ref_prefix(
                    nested,
                    old_ref=old_ref,
                    new_ref=new_ref,
                )

        elif isinstance(value, list):
            for nested in value:
                cls._replace_ref_prefix(
                    nested,
                    old_ref=old_ref,
                    new_ref=new_ref,
                )

    def _ensure_component_available(
        self,
        *,
        component_name: str,
        model: Any,
        mode: SchemaMode,
    ) -> None:
        existing_schema = self._schemas.get(component_name)
        if existing_schema is None:
            return

        for record in self._records.values():
            if record.component_name != component_name:
                continue

            if record.model is model and record.mode == mode:
                return

            raise SchemaCollisionError(
                f"Schema component {component_name!r} already exists "
                f"for a different model or schema mode."
            )

        raise SchemaCollisionError(
            f"Schema component {component_name!r} already exists."
        )

    @staticmethod
    def _default_component_name(
        *,
        base_name: str,
        mode: SchemaMode,
    ) -> str:
        if mode == "validation":
            return base_name
        return f"{base_name}.serialization"

    @staticmethod
    def _validate_mode(mode: SchemaMode) -> None:
        if mode not in ("validation", "serialization"):
            raise ValueError(f"Unsupported schema mode: {mode!r}")

    @staticmethod
    def _other_mode(mode: SchemaMode) -> SchemaMode:
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

        Every local $defs reference is rewritten to the final OpenAPI
        component namespace. Nested $defs are handled recursively.
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

        replacements = {
            f"#/components/schemas/{key}": (
                "#/components/schemas/"
                f"{self._nested_component_name(component_name, key)}"
            )
            for key in definitions
        }

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
            raise ValueError(
                "nested_naming_strategy must return a non-empty string"
            )

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