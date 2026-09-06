import pytest

from spectree.schema_registry import SchemaCollisionError, SchemaRegistry


def create_registry():
    return SchemaRegistry(
        naming_strategy=lambda model: model["name"],
        nested_naming_strategy=lambda parent, child: f"{parent}.{child}",
    )


def user_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
    }


def test_same_model_and_mode_is_registered_once():
    registry = create_registry()
    model = {"name": "User"}
    schema = user_schema()

    first = registry.register(model, "validation", schema)
    second = registry.register(model, "validation", schema)

    assert first == "User"
    assert second == "User"
    assert list(registry) == ["User"]


def test_same_schema_is_shared_between_validation_and_serialization():
    registry = create_registry()
    model = {"name": "User"}
    schema = user_schema()

    validation = registry.register(model, "validation", schema)
    serialization = registry.register(model, "serialization", schema)

    assert validation == "User"
    assert serialization == "User"
    assert list(registry) == ["User"]


def test_different_schemas_have_deterministic_component_names():
    registry = create_registry()
    model = {"name": "User"}

    validation_schema = user_schema()
    serialization_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "display_name": {"type": "string"},
        },
    }

    validation = registry.register(
        model,
        "validation",
        validation_schema,
    )
    serialization = registry.register(
        model,
        "serialization",
        serialization_schema,
    )

    assert validation == "User"
    assert serialization == "User.serialization"

    assert set(registry) == {
        "User",
        "User.serialization",
    }


def test_different_schemas_are_deterministic_when_serialization_is_registered_first():
    registry = create_registry()
    model = {"name": "User"}

    validation_schema = user_schema()
    serialization_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "display_name": {"type": "string"},
        },
    }

    serialization = registry.register(
        model,
        "serialization",
        serialization_schema,
    )
    validation = registry.register(
        model,
        "validation",
        validation_schema,
    )

    assert validation == "User"
    assert serialization == "User.serialization"

    assert set(registry) == {
        "User",
        "User.serialization",
    }


def test_different_models_with_same_name_raise():
    registry = create_registry()

    first_model = {"name": "User"}
    second_model = {"name": "User"}

    schema = {"type": "object"}

    registry.register(
        first_model,
        "validation",
        schema,
    )

    with pytest.raises(SchemaCollisionError):
        registry.register(
            second_model,
            "validation",
            schema,
        )


def test_different_models_with_same_name_across_modes_raise():
    registry = create_registry()

    first_model = {"name": "User"}
    second_model = {"name": "User"}

    schema = {"type": "object"}

    registry.register(
        first_model,
        "validation",
        schema,
    )

    with pytest.raises(SchemaCollisionError):
        registry.register(
            second_model,
            "serialization",
            schema,
        )


def test_nested_refs_use_final_component_name():
    registry = create_registry()

    model = {"name": "User"}

    schema = {
        "type": "object",
        "properties": {
            "profile": {
                "$ref": "#/components/schemas/Profile",
            },
        },
        "$defs": {
            "Profile": {
                "type": "object",
            },
        },
    }

    component_name = registry.register(
        model,
        "validation",
        schema,
    )

    assert component_name == "User"
    assert registry["User"]["properties"]["profile"]["$ref"] == (
        "#/components/schemas/User.Profile"
    )


def test_nested_schema_refs_are_normalized_recursively():
    registry = create_registry()

    model = {"name": "User"}

    schema = {
        "type": "object",
        "properties": {
            "profile": {
                "$ref": "#/components/schemas/Profile",
            },
        },
        "$defs": {
            "Profile": {
                "type": "object",
                "properties": {
                    "address": {
                        "$ref": "#/components/schemas/Address",
                    },
                },
                "$defs": {
                    "Address": {
                        "type": "object",
                    },
                },
            },
        },
    }

    registry.register(model, "validation", schema)

    registered = registry["User"]

    assert registered["properties"]["profile"]["$ref"] == (
        "#/components/schemas/User.Profile"
    )

    assert registered["$defs"]["Profile"]["properties"]["address"]["$ref"] == (
        "#/components/schemas/User.Profile.Address"
    )


def test_schemas_are_not_aliased_to_input():
    registry = create_registry()

    model = {"name": "User"}
    schema = user_schema()

    registry.register(model, "validation", schema)

    schema["properties"]["name"]["description"] = "changed"

    assert "description" not in registry["User"]["properties"]["name"]


def test_registered_schema_cannot_be_mutated_through_mapping_access():
    registry = create_registry()

    model = {"name": "User"}
    registry.register(model, "validation", user_schema())

    registered = registry["User"]
    registered["properties"]["name"]["description"] = "changed"

    assert "description" not in registry["User"]["properties"]["name"]


def test_snapshot_is_detached_from_registry():
    registry = create_registry()

    model = {"name": "User"}
    registry.register(model, "validation", user_schema())

    snapshot = registry.snapshot()
    snapshot["User"]["properties"]["name"]["description"] = "changed"

    assert "description" not in registry["User"]["properties"]["name"]


def test_invalid_schema_mode_is_rejected():
    registry = create_registry()

    with pytest.raises(ValueError):
        registry.register(
            {"name": "User"},
            "invalid",  # type: ignore[arg-type]
            user_schema(),
        )


def test_re_registering_same_model_with_changed_schema_raises():
    registry = create_registry()

    model = {"name": "User"}

    registry.register(
        model,
        "validation",
        user_schema(),
    )

    changed = {
        "type": "object",
        "properties": {
            "name": {"type": "integer"},
        },
    }

    with pytest.raises(SchemaCollisionError):
        registry.register(
            model,
            "validation",
            changed,
        )


def test_serialization_suffix_collision_is_reported():
    registry = create_registry()

    user = {"name": "User"}
    unrelated = {"name": "User.serialization"}

    registry.register(
        unrelated,
        "validation",
        {"type": "string"},
    )

    registry.register(
        user,
        "validation",
        user_schema(),
    )

    with pytest.raises(SchemaCollisionError):
        registry.register(
            user,
            "serialization",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "display_name": {"type": "string"},
                },
            },
        )


def test_registration_order_produces_identical_registry_state():
    model = {"name": "User"}

    validation_schema = user_schema()
    serialization_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "display_name": {"type": "string"},
        },
    }

    first = create_registry()
    first.register(model, "validation", validation_schema)
    first.register(model, "serialization", serialization_schema)

    second = create_registry()
    second.register(model, "serialization", serialization_schema)
    second.register(model, "validation", validation_schema)

    assert first.snapshot() == second.snapshot()
    assert tuple(first) == tuple(second)


def test_renamed_serialization_component_updates_nested_refs():
    registry = create_registry()

    model = {"name": "User"}

    serialization_schema = {
        "type": "object",
        "properties": {
            "profile": {
                "$ref": "#/components/schemas/Profile",
            },
        },
        "$defs": {
            "Profile": {
                "type": "object",
            },
        },
    }

    validation_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
    }

    serialization = registry.register(
        model,
        "serialization",
        serialization_schema,
    )

    assert serialization == "User.serialization"

    validation = registry.register(
        model,
        "validation",
        validation_schema,
    )

    assert validation == "User"

    assert registry[
        "User.serialization"
    ]["properties"]["profile"]["$ref"] == (
        "#/components/schemas/User.serialization.Profile"
    )


def test_raw_schema_is_preserved_internally():
    registry = create_registry()

    model = {"name": "User"}
    schema = {
        "type": "object",
        "properties": {
            "profile": {
                "$ref": "#/components/schemas/Profile",
            },
        },
        "$defs": {
            "Profile": {
                "type": "object",
            },
        },
    }

    registry.register(model, "validation", schema)

    stored = next(
        record
        for record in registry._records.values()
        if record.model is model
    )

    assert stored.raw_schema["properties"]["profile"]["$ref"] == (
        "#/components/schemas/Profile"
    )

    assert stored.schema["properties"]["profile"]["$ref"] == (
        "#/components/schemas/User.Profile"
    )
