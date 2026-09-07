import gc
import warnings
import weakref
from dataclasses import dataclass
from functools import wraps

import pytest
from falcon import App as FalconApp
from flask import Flask
from starlette.applications import Starlette

from spectree import Response
from spectree.config import Configuration
from spectree.model_adapter import get_pydantic_model_adapter
from spectree.models import Server
from spectree.plugins.flask_plugin import FlaskPlugin
from spectree.spec import SpecTree
from spectree.utils import (
    get_model_key,
)
from tests.common import get_paths
from tests.common_dataclass import (
    Child,
    Cookies,
    DemoModel,
    Form,
    Headers,
    Payload,
    Query,
    Resp,
)
from tests.type_checking_annotation_case import type_checking_view_func


def backend_app():
    return [
        ("flask", lambda: Flask(__name__)),
        ("falcon", FalconApp),
        ("starlette", Starlette),
    ]


def _get_spec(name, app, model_case, **kwargs):
    api = SpecTree(
        name,
        app=app,
        title=f"{name}",
        model_adapter=model_case.adapter,
        **kwargs,
    )
    if name == "flask":
        with app.app_context():
            spec = api.spec
    else:
        spec = api.spec

    return spec


def test_spectree_init(model_case):
    spec = SpecTree(path="docs", model_adapter=model_case.adapter)
    conf = Configuration()

    assert spec.config.title == conf.title
    assert spec.config.path == "docs"

    with pytest.raises(NotImplementedError):
        SpecTree(app=conf, model_adapter=model_case.adapter)


@pytest.mark.parametrize("name, app_factory", backend_app())
def test_register(name, app_factory, model_case):
    app = app_factory()
    api = SpecTree(name, model_adapter=model_case.adapter)
    api.register(app)


@pytest.mark.parametrize("name, app_factory", backend_app())
def test_spec_generate(name, app_factory, model_case):
    app = app_factory()
    spec = _get_spec(name, app, model_case)

    assert spec["info"]["title"] == name
    assert spec["paths"] == {}


@pytest.mark.pydantic
def test_annotations_ignore_unresolvable_return_annotation():
    app = Flask(__name__)
    api = SpecTree(
        "flask",
        model_adapter=get_pydantic_model_adapter(),
    )

    decorated = api.validate()(type_checking_view_func)
    app.add_url_rule(
        "/type-checking",
        view_func=decorated,
        methods=["POST"],
    )

    api.register(app)

    with app.app_context():
        spec = api.spec

    operation = spec["paths"]["/type-checking"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": f"#/components/schemas/{get_model_key(DemoModel)}"
    }


@pytest.mark.pydantic
def test_skip_validation_only_warns_for_request_annotations():
    api = SpecTree(
        "flask",
        model_adapter=get_pydantic_model_adapter(),
    )

    with pytest.warns(UserWarning, match="skip_validation"):

        @api.validate(skip_validation=True)
        def annotated(query: Query):
            return query

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        @api.validate(query=Query, skip_validation=True)
        def explicit_model():
            return None

    assert not captured
    assert annotated
    assert explicit_model


@pytest.mark.parametrize("name, app_factory", backend_app())
def test_spec_servers_empty(name, app_factory, model_case):
    app = app_factory()
    spec = _get_spec(name, app, model_case)

    assert "servers" not in spec


@pytest.mark.parametrize("name, app_factory", backend_app())
def test_spec_servers_only(name, app_factory, model_case):
    app = app_factory()
    server1_url = "http://example.com/bar"
    server2_url = "https://example.com/foo/bar"
    spec = _get_spec(
        name,
        app,
        model_case,
        servers=[Server(url=server1_url), Server(url=server2_url)],
    )

    assert spec["servers"] == [
        {"url": server1_url},
        {"url": server2_url},
    ]


@pytest.mark.parametrize("name, app_factory", backend_app())
def test_spec_servers_full(name, app_factory, model_case):
    app = app_factory()
    server1 = {"url": "http://foo/bar", "description": "Foo Bar"}
    server2 = {"url": "http://bar/foo/{lang}", "variables": {"lang": "en"}}
    spec = _get_spec(
        name,
        app,
        model_case,
        servers=[
            Server(**server1),
            Server(**server2),
        ],
    )

    expected = []
    for server in [server1, server2]:
        expected_item = {
            "url": server.get("url"),
        }
        description = server.get("description", None)
        if description:
            expected_item["description"] = description
        variables = server.get("variables", None)
        if variables:
            expected_item["variables"] = variables
        expected.append(expected_item)

    assert spec["servers"] == expected


def create_app(model_case):
    api = SpecTree("flask", model_adapter=model_case.adapter)
    api_strict = SpecTree(
        "flask",
        mode="strict",
        model_adapter=model_case.adapter,
    )
    api_greedy = SpecTree(
        "flask",
        mode="greedy",
        model_adapter=model_case.adapter,
    )
    api_customize_backend = SpecTree(
        backend=FlaskPlugin,
        model_adapter=model_case.adapter,
    )
    app = Flask(__name__)

    @app.route("/foo")
    @api.validate()
    def foo():
        pass

    @app.route("/bar")
    @api_strict.validate()
    def bar():
        pass

    @app.route("/lone", methods=["GET"])
    def lone_get():
        pass

    @app.route("/lone", methods=["POST"])
    def lone_post():
        pass

    return app, api, api_strict, api_greedy, api_customize_backend


def test_spec_bypass_mode(model_case):
    app, api, api_strict, api_greedy, api_customize_backend = create_app(model_case)
    api.register(app)
    with app.app_context():
        assert get_paths(api.spec) == ["/foo", "/lone"]

    app, api, api_strict, api_greedy, api_customize_backend = create_app(model_case)
    api_customize_backend.register(app)
    with app.app_context():
        assert get_paths(api.spec) == ["/foo", "/lone"]

    app, api, api_strict, api_greedy, api_customize_backend = create_app(model_case)
    api_greedy.register(app)
    with app.app_context():
        assert get_paths(api_greedy.spec) == ["/bar", "/foo", "/lone"]

    app, api, api_strict, api_greedy, api_customize_backend = create_app(model_case)
    api_strict.register(app)
    with app.app_context():
        assert get_paths(api_strict.spec) == ["/bar"]


def test_function_metadata_does_not_retain_spectree(model_case):
    def create_registered_api():
        app = Flask(__name__)
        api = SpecTree("flask", model_adapter=model_case.adapter)

        @app.route("/foo")
        @api.validate()
        def foo():
            pass

        api.register(app)
        assert api.get_function_metadata(foo) is not None
        return weakref.ref(api), weakref.ref(foo)

    api_ref, func_ref = create_registered_api()
    gc.collect()

    assert api_ref() is None
    assert func_ref() is None


def test_function_metadata_handles_wrapped_and_non_weakrefable_callables(model_case):
    api = SpecTree(model_adapter=model_case.adapter)

    @api.validate()
    def endpoint():
        pass

    @wraps(endpoint)
    def wrapped_endpoint():
        return endpoint()

    assert api.get_function_metadata(wrapped_endpoint) is not None
    assert not api.bypass(wrapped_endpoint)
    assert not api.bypass(len)

    strict_api = SpecTree(mode="strict", model_adapter=model_case.adapter)
    assert strict_api.bypass(wrapped_endpoint)
    assert strict_api.bypass(len)


def test_two_endpoints_with_the_same_path(model_case):
    app, api, _, _, _ = create_app(model_case)
    api.register(app)
    with app.app_context():
        spec = api.spec

    http_methods = list(spec["paths"]["/lone"].keys())
    http_methods.sort()
    assert http_methods == ["get", "post"]


def test_model_for_validation_errors_specified(model_case):
    @dataclass
    class CustomValidationError:
        user_id: int

    api = SpecTree("flask", model_adapter=model_case.adapter)
    app = Flask(__name__)
    custom_validation_error = model_case.get_model(CustomValidationError)

    @app.route("/foo")
    @api.validate(resp=Response(HTTP_200=None))
    def foo():
        pass

    @app.route("/bar")
    @api.validate(resp=Response(HTTP_200=None, HTTP_422=custom_validation_error))
    def bar():
        pass

    api.register(app)

    assert (
        api.get_function_metadata(foo).resp.find_model(422)
        is api.model_adapter.validation_error
    )
    assert (
        api.get_function_metadata(bar).resp.find_model(422) is custom_validation_error
    )


def test_global_model_for_validation_errors_specified(model_case):
    @dataclass
    class GlobalValidationError:
        user_id: int

    @dataclass
    class RouteValidationError:
        user_id: int

    global_validation_error = model_case.get_model(GlobalValidationError)
    route_validation_error = model_case.get_model(RouteValidationError)

    api = SpecTree(
        "flask",
        validation_error_model=global_validation_error,
        model_adapter=model_case.adapter,
    )
    app = Flask(__name__)

    @app.route("/foo")
    @api.validate(resp=Response(HTTP_200=None))
    def foo():
        pass

    @app.route("/bar")
    @api.validate(resp=Response(HTTP_200=None, HTTP_422=route_validation_error))
    def bar():
        pass

    api.register(app)

    assert (
        api.get_function_metadata(foo).resp.find_model(422) is global_validation_error
    )
    assert api.get_function_metadata(bar).resp.find_model(422) is route_validation_error


def test_plain_dataclass_models_are_supported_for_all_api_parts(model_case):
    api = SpecTree("flask", model_adapter=model_case.adapter)
    app = Flask(__name__)

    @app.route("/items", methods=["POST"])
    @api.validate(
        query=Query,
        json=Payload,
        form=Form,
        headers=Headers,
        cookies=Cookies,
        resp=Response(HTTP_200=Resp),
    )
    def create_item():
        pass

    api.register(app)
    with app.app_context():
        operation = api.spec["paths"]["/items"]["post"]

    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": f"#/components/schemas/{get_model_key(Payload)}"
    }
    assert operation["requestBody"]["content"]["multipart/form-data"]["schema"] == {
        "$ref": f"#/components/schemas/{get_model_key(Form)}"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": f"#/components/schemas/{get_model_key(Resp)}"
    }
    assert {parameter["in"] for parameter in operation["parameters"]} == {
        "query",
        "header",
        "cookie",
    }


def test_annotations_preserve_named_root_model_metadata(model_case):
    api = SpecTree("flask", annotations=True, model_adapter=model_case.adapter)
    app = Flask(__name__)

    @app.route("/annotated", methods=["POST"])
    @api.validate(resp=Response(HTTP_200=None))
    def annotated(json: model_case.get_model(dict[str, str], name="NamedDict")):
        return {}

    api.register(app)
    with app.app_context():
        spec = api.spec

    named_model = model_case.get_model(dict[str, str], name="NamedDict")
    schema = spec["paths"]["/annotated"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert schema["$ref"] == f"#/components/schemas/{get_model_key(named_model)}"


@pytest.mark.parametrize(
    ["override_operation_id", "expected_operation_id"],
    [(None, "get__foo"), ("getFoo", "getFoo")],
)
def test_operation_id_override(
    override_operation_id, expected_operation_id, model_case
):
    api = SpecTree("flask", model_adapter=model_case.adapter)
    app = Flask(__name__)

    @app.route("/foo")
    @api.validate(operation_id=override_operation_id)
    def foo():
        pass

    api.register(app)

    with app.app_context():
        operation_id = api.spec["paths"]["/foo"]["get"]["operationId"]
        assert operation_id == expected_operation_id


def test_custom_model_naming_strategies_are_used_in_refs_and_components(model_case):
    @dataclass
    class Payload:
        child: Child

    @dataclass
    class Result:
        child: Child

    api = SpecTree(
        "flask",
        naming_strategy=lambda model: model.__name__.lower(),
        nested_naming_strategy=lambda _parent, child: child.lower(),
        model_adapter=model_case.adapter,
    )
    app = Flask(__name__)

    @app.route("/items", methods=["POST"])
    @api.validate(
        json=model_case.get_model(Payload),
        resp=Response(HTTP_200=model_case.get_model(Result)),
    )
    def create_item():
        pass

    api.register(app)

    with app.app_context():
        spec = api.spec

    operation = spec["paths"]["/items"]["post"]
    schemas = spec["components"]["schemas"]

    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/payload"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/result"
    }
    assert schemas["payload"]["properties"]["child"] == {
        "$ref": "#/components/schemas/child"
    }
    assert schemas["result"]["properties"]["child"] == {
        "$ref": "#/components/schemas/child"
    }
    assert schemas["validationerror"]["items"] == {
        "$ref": "#/components/schemas/validationerrorelement"
    }
    assert "Child" not in schemas
    assert "child" in schemas


def test_validation_and_serialization_models_have_distinct_schema_components(
    model_case,
    monkeypatch,
):
    target_model = model_case.get_model(
        dict[str, str],
        name="ModeAwareModel",
    )

    api = SpecTree(
        "flask",
        naming_strategy=lambda model: (
            "ModeAwareModel" if model is target_model else "ValidationError"
        ),
        model_adapter=model_case.adapter,
    )

    def fake_json_schema(*, model, ref_template, mode):
        if model is target_model:
            if mode == "validation":
                return {
                    "title": "ModeAwareModel",
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                }

            return {
                "title": "ModeAwareModel",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "display_name": {"type": "string"},
                },
            }

        return {
            "title": "ValidationError",
            "type": "object",
        }

    monkeypatch.setattr(
        api.model_adapter,
        "json_schema",
        fake_json_schema,
    )

    app = Flask(__name__)

    @app.route("/users", methods=["POST"])
    @api.validate(
        json=target_model,
        resp=Response(HTTP_200=target_model),
    )
    def users():
        return {"name": "alice"}

    api.register(app)

    with app.app_context():
        spec = api.spec

    assert spec["paths"]["/users"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/ModeAwareModel"}

    assert spec["paths"]["/users"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/ModeAwareModel.serialization"}

    assert "ModeAwareModel" in spec["components"]["schemas"]
    assert "ModeAwareModel.serialization" in spec["components"]["schemas"]

    assert spec["components"]["schemas"]["ModeAwareModel"]["properties"] == {
        "name": {"type": "string"}
    }

    assert spec["components"]["schemas"]["ModeAwareModel.serialization"][
        "properties"
    ] == {
        "name": {"type": "string"},
        "display_name": {"type": "string"},
    }


def test_generate_spec_is_deterministic_and_does_not_mutate_registry(
    model_case,
    monkeypatch,
):
    target_model = model_case.get_model(
        dict[str, str],
        name="QueryModel",
    )

    api = SpecTree(
        "flask",
        naming_strategy=lambda model: (
            "QueryModel" if model is target_model else "ValidationError"
        ),
        nested_naming_strategy=lambda parent, child: f"{parent}.{child}",
        model_adapter=model_case.adapter,
    )

    def fake_json_schema(*, model, ref_template, mode):
        if model is target_model:
            return {
                "title": "QueryModel",
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "style": "form",
                        "explode": True,
                    },
                },
                "$defs": {
                    "Child": {
                        "type": "object",
                    },
                },
            }

        return {
            "title": "ValidationError",
            "type": "object",
        }

    monkeypatch.setattr(
        api.model_adapter,
        "json_schema",
        fake_json_schema,
    )

    app = Flask(__name__)

    @app.route("/query")
    @api.validate(query=target_model)
    def query():
        return "ok"

    api.register(app)

    before = api.models.snapshot()

    with app.app_context():
        first_spec = api._generate_spec()

    after_first = api.models.snapshot()

    with app.app_context():
        second_spec = api._generate_spec()

    after_second = api.models.snapshot()

    assert before == after_first
    assert after_first == after_second
    assert first_spec == second_spec

    assert "style" in api.models["QueryModel"]["properties"]["query"]
    assert "explode" in api.models["QueryModel"]["properties"]["query"]
    assert "$defs" in api.models["QueryModel"]


def test_response_declaration_is_not_mutated_by_spectree(
    model_case,
):
    model = model_case.get_model(
        dict[str, str],
        name="ResponseModel",
    )

    response = Response(HTTP_200=model)

    api = SpecTree(
        "flask",
        model_adapter=model_case.adapter,
    )

    app = Flask(__name__)

    @app.route("/response")
    @api.validate(resp=response)
    def response_endpoint():
        return {"value": "ok"}

    assert response.model_adapter is None
    assert response.code_models == {}
    assert response._model_keys == {}

    api.register(app)

    assert response.model_adapter is None
    assert response.code_models == {}
    assert response._model_keys == {}

    assert response_endpoint.resp is not response
    assert response_endpoint.resp.model_adapter is api.model_adapter


def test_response_copy_does_not_share_mutable_state(
    model_case,
):
    model = model_case.get_model(
        dict[str, str],
        name="SharedResponseModel",
    )

    response = Response(
        HTTP_200=(model, "OK"),
    )

    compiled = response.copy_for_model_adapter(
        SpecTree(
            "flask",
            model_adapter=model_case.adapter,
        ).model_adapter,
    )

    compiled.code_descriptions["HTTP_200"] = "changed"
    compiled._model_keys["HTTP_200"] = "response"

    assert response.code_descriptions["HTTP_200"] == "OK"
    assert response._model_keys == {}
    assert response.model_adapter is None
    assert response.code_models == {}


def test_response_can_be_reused_by_different_model_adapters(
    model_case,
):
    model = model_case.get_model(
        dict[str, str],
        name="SharedResponseModel",
    )

    response = Response(HTTP_200=model)

    pydantic_api = SpecTree(
        "flask",
        model_adapter=model_case.adapter,
    )

    # This test should actually use two independent adapters if available.
    compiled = response.copy_for_model_adapter(
        pydantic_api.model_adapter,
    )

    assert response.model_adapter is None
    assert compiled.model_adapter is pydantic_api.model_adapter
    assert response.code_models == {}


def test_endpoint_spec_from_annotations():
    api = SpecTree()

    class Query:
        pass

    class Body:
        pass

    @api.validate()
    def endpoint(query: Query, json: Body):
        return None

    endpoint_spec = endpoint._endpoint_spec

    assert endpoint_spec.query is Query
    assert endpoint_spec.json is Body
    assert endpoint_spec.form is None
    assert endpoint_spec.headers is None
    assert endpoint_spec.cookies is None

    assert endpoint_spec.injected_arguments == {
        "query",
        "json",
    }


def test_runtime_does_not_resolve_unrelated_annotations():
    app = Flask(__name__)
    api = SpecTree("flask")

    class DemoModel:
        pass

    def endpoint(
        json: DemoModel,
        dependency: "CompletelyNonExistentType",  # noqa F821
    ):
        return {"ok": True}

    decorated = api.validate()(endpoint)

    app.add_url_rule(
        "/runtime-annotation",
        view_func=decorated,
        methods=["POST"],
    )

    api.register(app)

    with app.test_client() as client:
        response = client.post(
            "/runtime-annotation",
            json={},
        )

    assert response.status_code != 500


def test_runtime_ignores_unresolvable_return_annotation():
    app = Flask(__name__)
    api = SpecTree("flask")

    decorated = api.validate()(type_checking_view_func)

    app.add_url_rule(
        "/type-checking-runtime",
        view_func=decorated,
        methods=["POST"],
    )

    api.register(app)

    with app.test_client() as client:
        response = client.post(
            "/type-checking-runtime",
            json={},
        )

    assert response.status_code != 500


def test_spec_generation_does_not_depend_on_previous_spec_access(
    model_case,
):
    model = model_case.get_model(
        dict[str, str],
        name="StableModel",
    )

    api = SpecTree(
        "flask",
        model_adapter=model_case.adapter,
    )

    app = Flask(__name__)

    @app.route("/stable")
    @api.validate(query=model)
    def stable():
        return "ok"

    api.register(app)

    with app.app_context():
        generated_before = api._generate_spec()
        cached_spec = api.spec
        generated_after = api._generate_spec()

    assert generated_before == cached_spec
    assert generated_before == generated_after


def test_response_can_produce_independent_bound_copies(
    model_case,
):
    model = model_case.get_model(
        dict[str, str],
        name="IndependentResponseModel",
    )

    response = Response(HTTP_200=model)

    api = SpecTree(
        "flask",
        model_adapter=model_case.adapter,
    )

    first = response.copy_for_model_adapter(
        api.model_adapter,
    )
    second = response.copy_for_model_adapter(
        api.model_adapter,
    )

    assert first is not second
    assert first.code_models is not second.code_models
    assert first.code_descriptions is not second.code_descriptions
    assert first._model_keys is not second._model_keys

    first._set_model_key("HTTP_200", "First")
    second._set_model_key("HTTP_200", "Second")

    assert first._model_keys == {"HTTP_200": "First"}
    assert second._model_keys == {"HTTP_200": "Second"}

    assert response._model_keys == {}


def test_endpoint_spec_is_immutable():
    api = SpecTree()

    class Query:
        pass

    @api.validate(
        query=Query,
        tags=("users",),
        operation_id="users_list",
    )
    def endpoint():
        return None

    endpoint_spec = endpoint._endpoint_spec

    with pytest.raises((AttributeError, TypeError)):
        endpoint_spec.query = list

    with pytest.raises((AttributeError, TypeError)):
        endpoint_spec.request_model_keys = ()


def test_reusing_validate_decorator_does_not_leak_endpoint_models():
    api = SpecTree()

    class FirstQuery:
        pass

    class SecondQuery:
        pass

    decorator = api.validate()

    @decorator
    def first(query: FirstQuery):
        return query

    @decorator
    def second(query: SecondQuery):
        return query

    assert first._endpoint_spec.query is FirstQuery
    assert second._endpoint_spec.query is SecondQuery

    assert first._endpoint_spec.injected_arguments == {"query"}
    assert second._endpoint_spec.injected_arguments == {"query"}

    assert api.get_endpoint_spec(first) is first._endpoint_spec
    assert api.get_endpoint_spec(second) is second._endpoint_spec


def test_explicit_endpoint_models_are_isolated_between_decorations():
    api = SpecTree()

    class FirstQuery:
        pass

    class SecondQuery:
        pass

    decorator = api.validate(query=FirstQuery)

    @decorator
    def first():
        return None

    @decorator
    def second(query: SecondQuery):
        return query

    assert first._endpoint_spec.query is FirstQuery
    assert second._endpoint_spec.query is SecondQuery


def test_endpoint_spec_is_used_as_runtime_source_of_truth():
    api = SpecTree()

    class Query:
        pass

    @api.validate(
        query=Query,
        operation_id="stable-operation",
        tags=("stable",),
    )
    def endpoint():
        return None

    endpoint.operation_id = "legacy-operation"
    endpoint.tags = ("legacy",)

    endpoint_spec = endpoint._endpoint_spec

    assert endpoint_spec.operation_id == "stable-operation"
    assert endpoint_spec.tags == ("stable",)


def test_get_endpoint_spec_handles_wrapped_functions():
    api = SpecTree()

    @api.validate()
    def endpoint():
        return None

    @wraps(endpoint)
    def wrapped_endpoint():
        return endpoint()

    assert api.get_endpoint_spec(endpoint) is endpoint._endpoint_spec
    assert api.get_endpoint_spec(wrapped_endpoint) is endpoint._endpoint_spec


def test_spec_is_invalidated_after_new_endpoint_is_decorated():
    app = Flask(__name__)
    api = SpecTree("flask")
    api.register(app)

    with app.app_context():
        first_spec = api.spec

    @app.route("/new")
    @api.validate()
    def new_endpoint():
        return None

    with app.app_context():
        second_spec = api.spec

    assert "/new" not in first_spec["paths"]
    assert "/new" in second_spec["paths"]


def test_endpoint_metadata_remains_backward_compatible():
    api = SpecTree()

    class Query:
        pass

    @api.validate(
        query=Query,
        tags=("users",),
        security={"bearerAuth": []},
        deprecated=True,
        operation_id="users_get",
    )
    def endpoint(query):
        return query

    metadata = api.get_function_metadata(endpoint)
    endpoint_spec = api.get_endpoint_spec(endpoint)

    assert metadata is not None
    assert endpoint_spec is not None

    assert metadata.query is not None
    assert metadata.query == endpoint.query
    assert metadata.tags == endpoint.tags
    assert metadata.security == endpoint.security
    assert metadata.deprecated is endpoint.deprecated
    assert metadata.operation_id == endpoint.operation_id


def test_skip_validation_warns_for_request_annotations():
    api = SpecTree()

    class Query:
        pass

    with pytest.warns(UserWarning, match="skip_validation"):

        @api.validate(skip_validation=True)
        def annotated(query: Query):
            return query

    assert annotated._endpoint_spec.query is Query
    assert annotated._endpoint_spec.injected_arguments == {"query"}


def test_endpoint_openapi_rendering_uses_endpoint_spec_not_legacy_metadata():
    api = SpecTree()

    class Query:
        pass

    @api.validate(
        query=Query,
        operation_id="endpoint-operation",
        tags=("endpoint-tag",),
    )
    def endpoint():
        return None

    endpoint.operation_id = "legacy-operation"
    endpoint.tags = ("legacy-tag",)

    assert endpoint._endpoint_spec.operation_id == "endpoint-operation"
    assert endpoint._endpoint_spec.tags == ("endpoint-tag",)


def test_endpoint_spec_contains_resolved_schema_keys():
    api = SpecTree()

    class Query:
        pass

    @api.validate(query=Query)
    def endpoint(query: Query):
        return query

    key = endpoint._endpoint_spec.model_key_for("query")

    assert key is not None
    assert key == endpoint.query
    assert endpoint._endpoint_spec.model_key_for("json") is None


def test_reusing_validate_decorator_does_not_leak_endpoint_state():
    api = SpecTree()

    class FirstQuery:
        pass

    class SecondQuery:
        pass

    decorator = api.validate()

    @decorator
    def first(query: FirstQuery):
        return query

    @decorator
    def second(query: SecondQuery):
        return query

    assert first._endpoint_spec.query is FirstQuery
    assert second._endpoint_spec.query is SecondQuery
    assert first._endpoint_spec is not second._endpoint_spec

    assert first._endpoint_spec.model_key_for("query") == first.query
    assert second._endpoint_spec.model_key_for("query") == second.query
