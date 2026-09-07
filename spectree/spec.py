import warnings
import weakref
from collections import defaultdict
from functools import wraps
from importlib import import_module
from typing import Any, Callable, Mapping, Sequence

from spectree._types import (
    HookHandler,
    ModelAdapterType,
    NamingStrategy,
    NestedNamingStrategy,
)
from spectree.config import Configuration, ModeEnum
from spectree.endpoint import REQUEST_MODEL_ARGUMENTS, EndpointSpec
from spectree.metadata import (
    FunctionDecorator,
    is_validated_function,
    iter_wrapped_functions,
    register_validated_function,
)
from spectree.model_adapter import ModelSpec, get_pydantic_model_adapter
from spectree.model_adapter.protocol import SchemaMode
from spectree.models import Tag
from spectree.plugins import PLUGINS, BasePlugin
from spectree.response import Response
from spectree.schema_registry import SchemaRegistry
from spectree.utils import (
    default_after_handler,
    default_before_handler,
    get_model_key,
    get_nested_key,
    get_request_model_hints,
    get_security,
    json_compatible_deepcopy,
    parse_comments,
    parse_name,
)


class SpecTree:
    """
    Interface
    :param str backend_name: choose from
        ('flask', 'quart', 'falcon', 'falcon-asgi', 'starlette')
    :param backend: a backend that inherit `SpecTree.plugins.base.BasePlugin`, this will
        override the `backend_name` if provided
    :param app: backend framework application instance (can be registered later)
    :param before: a callback function of the form
        :meth:`spectree.utils.default_before_handler`
        ``func(req, resp, req_validation_error, instance, model_adapter)``
        that will be called after the request validation before the endpoint function
    :param after: a callback function of the form
        :meth:`spectree.utils.default_after_handler`
        ``func(req, resp, resp_validation_error, instance, model_adapter)``
        that will be called after the response validation
    :param validation_error_status: The default response status code to use in the
        event of a validation error. This value can be overridden for specific endpoints
        if needed.
    :param validation_error_model: The default validation error type to be shown
        in the generated OpenAPI frontend. Make sure it's derived from the model
        adapter (including the ValidationError).
    :param naming_strategy: A callable that receives a model type/expression and returns
        the top-level component schema name used in the OpenAPI document.
    :param nested_naming_strategy: A callable that receives ``(parent, child)``
        schema names and returns the component schema name for nested models.
    :param model_adapter: adapter for validation and OpenAPI JSON schema generation.
        Choose from the `spectree.model_adapter`. If not set, will use `pydantic`.
    :param kwargs: init :class:`spectree.config.Configuration`, they can also be
        configured through the environment variables with prefix `spectree_`
    """

    def __init__(
        self,
        backend_name: str = "base",
        backend: type[BasePlugin] | None = None,
        app: Any = None,
        before: HookHandler = default_before_handler,
        after: HookHandler = default_after_handler,
        validation_error_status: int = 422,
        validation_error_model: ModelSpec | None = None,
        naming_strategy: NamingStrategy = get_model_key,
        nested_naming_strategy: NestedNamingStrategy = get_nested_key,
        model_adapter: ModelAdapterType | None = None,
        **kwargs: Any,
    ):
        self.naming_strategy = naming_strategy
        self.nested_naming_strategy = nested_naming_strategy
        self.validation_error_status = validation_error_status
        self.model_adapter = model_adapter or get_pydantic_model_adapter()
        self.validation_error_model = validation_error_model
        self.before = before
        self.after = after

        self.config: Configuration = Configuration.model_validate(
            kwargs,
            model_adapter=self.model_adapter,
        )

        self.backend_name = backend_name

        if backend:
            self.backend = backend(self)
        else:
            plugin = PLUGINS[backend_name]
            module = import_module(plugin.name, plugin.package)
            self.backend = getattr(module, plugin.class_name)(self)

        self.models = SchemaRegistry(
            naming_strategy=self.naming_strategy,
            nested_naming_strategy=self.nested_naming_strategy,
        )

        self._function_metadata: weakref.WeakKeyDictionary[
            Callable, FunctionDecorator
        ] = weakref.WeakKeyDictionary()

        self._endpoint_specs: weakref.WeakKeyDictionary[Callable, EndpointSpec] = (
            weakref.WeakKeyDictionary()
        )

        if app:
            self.register(app)

    def register(self, app: Any):
        """
        register to backend application
        """
        self.app = app
        self.backend.register_route(self.app)

    @property
    def spec(self):
        """
        get the OpenAPI spec
        """
        if not hasattr(self, "_spec"):
            self._spec = self._generate_spec()
        return self._spec

    def bypass(self, func: Callable) -> bool:
        """
        Bypass rules for routes according to the configured mode.
        """
        if self.config.mode == ModeEnum.greedy:
            return False

        owned_by_self = self.get_endpoint_spec(func) is not None

        if self.config.mode == ModeEnum.strict:
            return not owned_by_self

        return not owned_by_self and is_validated_function(func)

    def get_endpoint_spec(self, func: Callable) -> EndpointSpec | None:
        """
        Return the compiled endpoint contract for a validated callable.
        """
        for candidate in iter_wrapped_functions(func):
            try:
                endpoint = self._endpoint_specs.get(candidate)
            except TypeError:
                continue

            if endpoint is not None:
                return endpoint

        return None

    def get_function_metadata(
        self,
        func: Callable,
    ) -> FunctionDecorator | None:
        """
        Return backward-compatible metadata for a callable validated by this
        SpecTree instance.
        """
        for candidate in iter_wrapped_functions(func):
            try:
                metadata = self._function_metadata.get(candidate)
            except TypeError:
                continue

            if metadata is not None:
                return metadata

        return None

    def validate(  # noqa: PLR0913, PLR0917
        self,
        query: ModelSpec | None = None,
        json: ModelSpec | None = None,
        form: ModelSpec | None = None,
        headers: ModelSpec | None = None,
        cookies: ModelSpec | None = None,
        resp: Response | None = None,
        tags: Sequence = (),
        security: Any = None,
        deprecated: bool = False,
        before: HookHandler | None = None,
        after: HookHandler | None = None,
        validation_error_status: int = 0,
        path_parameter_descriptions: Mapping[str, str] | None = None,
        skip_validation: bool = False,
        operation_id: str | None = None,
        force_resp_serialize: bool = False,
    ) -> Callable:
        """
        Validate request/response and compile an immutable endpoint contract.

        Request annotations are resolved exactly once during decoration.
        Runtime plugins consume the resulting ``EndpointSpec`` directly.
        """

        effective_validation_error_status = (
            self.validation_error_status
            if validation_error_status == 0
            else validation_error_status
        )

        def decorate_validation(func: Callable):
            resolved_annotations: Mapping[str, Any] = {}
            effective_models: dict[str, ModelSpec | None] = {
                "query": query,
                "json": json,
                "form": form,
                "headers": headers,
                "cookies": cookies,
            }

            if self.config.annotations:
                resolved_annotations = get_request_model_hints(func)

                if skip_validation and resolved_annotations:
                    warnings.warn(
                        "`skip_validation` cannot be used with `annotations` enabled. "
                        "The instances of `json`, `headers`, `cookies`, etc. read from "
                        "the function will be `None`.",
                        UserWarning,
                        stacklevel=2,
                    )

                for name in REQUEST_MODEL_ARGUMENTS:
                    if name in resolved_annotations:
                        effective_models[name] = resolved_annotations[name]

            injected_arguments = frozenset(
                name
                for name in REQUEST_MODEL_ARGUMENTS
                if name in resolved_annotations and effective_models[name] is not None
            )

            request_model_keys: dict[str, str] = {}

            for name in REQUEST_MODEL_ARGUMENTS:
                model = effective_models[name]
                if model is None:
                    continue

                request_model_keys[name] = self._add_model(
                    model=model,
                    mode="validation",
                )

            compiled_resp: Response | None = None

            if resp is not None:
                compiled_resp = resp.copy_for_model_adapter(
                    self.model_adapter,
                )

                compiled_resp.add_model(
                    effective_validation_error_status,
                    self.validation_error_model or self.model_adapter.validation_error,
                    replace=False,
                )

                for code, model in compiled_resp.code_models.items():
                    model_key = self._add_model(
                        model=model,
                        mode="serialization",
                    )
                    compiled_resp._set_model_key(code, model_key)

            endpoint = EndpointSpec(
                query=effective_models["query"],
                json=effective_models["json"],
                form=effective_models["form"],
                headers=effective_models["headers"],
                cookies=effective_models["cookies"],
                request_model_keys=tuple(
                    (name, request_model_keys[name])
                    for name in REQUEST_MODEL_ARGUMENTS
                    if name in request_model_keys
                ),
                response=compiled_resp,
                injected_arguments=injected_arguments,
                before=before if before is not None else self.before,
                after=after if after is not None else self.after,
                validation_error_status=effective_validation_error_status,
                skip_validation=skip_validation,
                force_resp_serialize=force_resp_serialize,
                tags=tuple(tags),
                security=security,
                deprecated=deprecated,
                path_parameter_descriptions=(
                    dict(path_parameter_descriptions)
                    if path_parameter_descriptions is not None
                    else None
                ),
                operation_id=operation_id,
            )

            metadata = FunctionDecorator(
                query=request_model_keys.get("query"),
                json=request_model_keys.get("json"),
                form=request_model_keys.get("form"),
                headers=request_model_keys.get("headers"),
                cookies=request_model_keys.get("cookies"),
                resp=compiled_resp,
                tags=endpoint.tags,
                security=endpoint.security,
                deprecated=endpoint.deprecated,
                path_parameter_descriptions=endpoint.path_parameter_descriptions,
                operation_id=endpoint.operation_id,
            )

            self._endpoint_specs[func] = endpoint
            self._function_metadata[func] = metadata

            register_validated_function(func)

            if self.backend.ASYNC:

                @wraps(func)
                async def validation(*args: Any, **kwargs: Any):
                    return await self.backend.validate(
                        func,
                        endpoint,
                        *args,
                        **kwargs,
                    )

            else:

                @wraps(func)
                def validation(*args: Any, **kwargs: Any):
                    return self.backend.validate(
                        func,
                        endpoint,
                        *args,
                        **kwargs,
                    )

            for name, model_key in request_model_keys.items():
                setattr(validation, name, model_key)

            validation.resp = compiled_resp
            validation.tags = endpoint.tags
            validation.security = endpoint.security
            validation.deprecated = endpoint.deprecated
            validation.path_parameter_descriptions = (
                endpoint.path_parameter_descriptions
            )
            validation.operation_id = endpoint.operation_id
            validation._endpoint_spec = endpoint
            validation._decorator = self

            self._spec = None
            return validation

        return decorate_validation

    def _add_model(
        self,
        model: ModelSpec,
        mode: SchemaMode = "validation",
    ) -> str:
        """
        Register a model schema and return its OpenAPI component name.
        """
        schema = self.model_adapter.json_schema(
            model=model,
            ref_template="#/components/schemas/{model}",
            mode=mode,
        )

        return self.models.register(
            model=model,
            mode=mode,
            schema=schema,
        )

    def _parse_endpoint_params(
        self,
        endpoint: EndpointSpec,
        params: list[Mapping[str, Any]],
        models: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        attr_to_spec_key = {
            "query": "query",
            "headers": "header",
            "cookies": "cookie",
        }
        route_param_keywords = ("explode", "style", "allowReserved")

        for attr, position in attr_to_spec_key.items():
            model_key = endpoint.model_key_for(attr)
            if model_key is None:
                continue

            model = models[model_key]
            properties = model.get(
                "properties",
                {model.get("title"): model},
            )

            for name, property_schema in properties.items():
                schema = property_schema.copy()
                extra = {
                    kw: schema.pop(kw) for kw in route_param_keywords if kw in schema
                }

                params.append(
                    {
                        "name": name,
                        "in": position,
                        "schema": schema,
                        "required": name in model.get("required", []),
                        "description": schema.get("description", ""),
                        **extra,
                    }
                )

        return params

    def _parse_endpoint_request(
        self,
        endpoint: EndpointSpec,
    ) -> dict[str, Any]:
        content_items: dict[str, Any] = {}

        json_key = endpoint.model_key_for("json")
        if json_key is not None:
            content_items["application/json"] = {
                "schema": {
                    "$ref": f"#/components/schemas/{json_key}",
                }
            }

        form_key = endpoint.model_key_for("form")
        if form_key is not None:
            content_items["multipart/form-data"] = {
                "schema": {
                    "$ref": f"#/components/schemas/{form_key}",
                }
            }

        return (
            {
                "content": content_items,
                "required": True,
            }
            if content_items
            else {}
        )

    def _generate_spec(self) -> dict[str, Any]:
        """
        Generate the OpenAPI specification from compiled endpoint contracts.
        """
        models = json_compatible_deepcopy(dict(self.models))

        routes: dict[str, dict[str, Any]] = defaultdict(dict)
        tags: dict[str, Any] = {}

        for route in self.backend.find_routes():
            for method, func in self.backend.parse_func(route):
                if self.backend.bypass(func, method) or self.bypass(func):
                    continue

                endpoint = self.get_endpoint_spec(func)
                metadata = (
                    self.get_function_metadata(func) if endpoint is None else None
                )

                path_parameter_descriptions = (
                    endpoint.path_parameter_descriptions
                    if endpoint is not None
                    else metadata.path_parameter_descriptions
                    if metadata is not None
                    else None
                )

                path, parameters = self.backend.parse_path(
                    route,
                    path_parameter_descriptions,
                )

                name = parse_name(func)
                summary, desc = parse_comments(func)

                if endpoint is not None:
                    func_tags = endpoint.tags
                else:
                    func_tags = metadata.tags if metadata is not None else ()

                for tag in func_tags:
                    if str(tag) not in tags:
                        tags[str(tag)] = (
                            tag.to_dict(exclude_none=True)
                            if isinstance(tag, Tag)
                            else {"name": tag}
                        )

                if endpoint is not None:
                    operation_id = (
                        endpoint.operation_id
                        or f"{method.lower()}_{path.replace('/', '_')}"
                    )
                    endpoint_parameters = self._parse_endpoint_params(
                        endpoint,
                        parameters[:],
                        models,
                    )
                    responses = (
                        endpoint.response.generate_spec(self.naming_strategy)
                        if endpoint.response is not None
                        else {}
                    )
                    request_body = self._parse_endpoint_request(endpoint)
                else:
                    operation_id = self.backend.get_func_operation_id(
                        func,
                        path,
                        method,
                    )
                    endpoint_parameters = (
                        metadata.parse_params(
                            parameters[:],
                            models,
                        )
                        if metadata is not None
                        else []
                    )
                    responses = (
                        metadata.parse_resp(self.naming_strategy)
                        if metadata is not None
                        else {}
                    )
                    request_body = (
                        metadata.parse_request() if metadata is not None else {}
                    )

                operation: dict[str, Any] = {
                    "summary": summary or f"{name} <{method}>",
                    "operationId": operation_id,
                    "description": desc or "",
                    "tags": [str(x) for x in func_tags],
                    "parameters": endpoint_parameters,
                    "responses": responses,
                }

                if endpoint is not None:
                    security = endpoint.security
                    deprecated = endpoint.deprecated
                else:
                    security = metadata.security if metadata is not None else None
                    deprecated = metadata.deprecated if metadata is not None else False

                if security is not None:
                    operation["security"] = get_security(security)

                if deprecated:
                    operation["deprecated"] = deprecated

                if request_body:
                    operation["requestBody"] = request_body

                routes[path][method.lower()] = operation

        spec: dict[str, Any] = {
            "openapi": self.config.openapi_version,
            "info": self.config.openapi_info(),
            "tags": list(tags.values()),
            "paths": dict(routes),
            "components": {
                "schemas": {
                    **models,
                    **self._get_model_definitions(models),
                },
            },
        }

        if self.config.servers:
            spec["servers"] = [
                server.to_dict(exclude_none=True) for server in self.config.servers
            ]

        if self.config.security_schemes:
            spec["components"]["securitySchemes"] = {
                scheme.name: scheme.data.to_dict(exclude_none=True)
                for scheme in self.config.security_schemes
            }

        spec["security"] = get_security(self.config.security)

        return spec
