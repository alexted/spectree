import inspect
from collections import namedtuple
from contextvars import ContextVar
from functools import partial
from typing import Any

from starlette.convertors import CONVERTOR_TYPES
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import compile_path

from spectree._types import ModelAdapterType
from spectree.endpoint import EndpointSpec
from spectree.model_adapter import ModelSpec
from spectree.plugins.base import (
    BasePlugin,
)
from spectree.request_data import RequestData
from spectree.utils import get_multidict_items_starlette

METHODS = {"get", "post", "put", "patch", "delete"}
Route = namedtuple("Route", ["path", "methods", "func"])
_active_model_adapter: ContextVar[ModelAdapterType | None] = ContextVar(
    "spectree_starlette_model_adapter", default=None
)
_response_models: dict[object, ModelSpec] = {}


def _get_response_model(model_adapter: ModelAdapterType) -> ModelSpec:
    response_model = _response_models.get(model_adapter)
    if response_model is None:
        response_model = model_adapter.make_root_model(
            Any,
            name="_SpecTreeStarletteResponseModel",
        )
        _response_models[model_adapter] = response_model
    return response_model


def _get_starlette_response_model_adapter() -> ModelAdapterType:
    model_adapter = _active_model_adapter.get()
    if model_adapter is None:
        raise RuntimeError(
            "SpecTreeStarletteResponse must be rendered inside a SpecTree request"
        )
    return model_adapter


class SpecTreeStarletteResponse(JSONResponse):
    def render(self, content) -> bytes:
        adapter = _get_starlette_response_model_adapter()
        response_model = _get_response_model(adapter)
        self._model_class = content.__class__
        return adapter.dump_json(adapter.validate_obj(response_model, content))


class StarlettePlugin(BasePlugin):
    ASYNC = True

    def __init__(self, spectree):
        super().__init__(spectree)
        self.conv2type = {conv: typ for typ, conv in CONVERTOR_TYPES.items()}

    def register_route(self, app):
        app.add_route(
            self.config.spec_url,
            lambda request: JSONResponse(self.spectree.spec),
        )
        for ui in self.config.page_templates:
            app.add_route(
                f"/{self.config.path}/{ui}",
                lambda request, ui=ui: HTMLResponse(
                    self.config.page_templates[ui].format(
                        spec_url=self.config.filename,
                        spec_path=self.config.path,
                        **self.config.swagger_oauth2_config(),
                    )
                ),
            )

    async def get_request_data(
        self, request: Request, endpoint: EndpointSpec
    ) -> RequestData:
        has_data = request.method not in ("GET", "DELETE")
        content_type = request.headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()

        use_json = endpoint.json and has_data and media_type == "application/json"
        use_form = endpoint.form and has_data and media_type in self.FORM_MIMETYPE

        req_json = None
        if use_json:
            req_json = await request.json()
            if req_json is None:
                req_json = {}

        req_form = None
        if use_form:
            req_form = get_multidict_items_starlette(
                await request.form(),
                endpoint.form,
            )

        return RequestData(
            query=get_multidict_items_starlette(
                request.query_params,
                endpoint.query,
            ),
            json=req_json,
            form=req_form,
            headers=dict(request.headers),
            cookies=dict(request.cookies),
        )

    def validate(
        self,
        request,
        endpoint: EndpointSpec,
        kwargs: dict,
    ):
        request_data = RequestData()
        req_validation_error = None

        if not endpoint.skip_validation:
            try:
                request_data = await self._get_and_validate_request_data(
                    request,
                    endpoint,
                )
                self.set_request_data(request, request_data)
            except self.model_adapter.validation_error as exc:
                req_validation_error = exc

        self.inject_request_data(request_data, endpoint, kwargs)

        if req_validation_error is not None:
            return self.on_validation_error(
                request,
                endpoint,
                req_validation_error,
            )

        return None

    def find_routes(self):
        routes = []

        def parse_route(app, prefix=""):
            if not app.routes:
                return
            for route in app.routes:
                if route.path.startswith(f"/{self.config.path}"):
                    continue
                func = route.app
                if isinstance(func, partial):
                    try:
                        func = func.__wrapped__
                    except AttributeError as err:
                        self.logger.warning(
                            "failed to get the wrapped func %s: %s",
                            func,
                            err,
                        )
                if inspect.isclass(func):
                    for method in METHODS:
                        if getattr(func, method, None):
                            routes.append(
                                Route(
                                    f"{prefix}{route.path}",
                                    {method.upper()},
                                    getattr(func, method),
                                )
                            )
                elif inspect.isfunction(func):
                    routes.append(
                        Route(f"{prefix}{route.path}", route.methods, route.endpoint)
                    )
                else:
                    parse_route(route, prefix=f"{prefix}{route.path}")

        parse_route(self.spectree.app)
        return routes

    def bypass(self, func, method):
        return method in ["HEAD", "OPTIONS"]

    def parse_func(self, route):
        for method in route.methods or ["GET"]:
            yield method, route.func

    def parse_path(self, route, path_parameter_descriptions):
        _, path, variables = compile_path(route.path)
        parameters = []
        for name, conv in variables.items():
            typ = self.conv2type[conv]
            if typ == "int":
                schema = {"type": "integer", "format": "int32"}
            elif typ == "float":
                schema = {"type": "number", "format": "float"}
            elif typ == "path":
                schema = {"type": "string", "format": "path"}
            elif typ == "str":
                schema = {"type": "string"}
            else:
                schema = None
            description = (
                path_parameter_descriptions.get(name, "")
                if path_parameter_descriptions
                else ""
            )
            parameters.append(
                {
                    "name": name,
                    "in": "path",
                    "required": True,
                    "schema": schema,
                    "description": description,
                }
            )
        return path, parameters

    async def get_request_data(
        self,
        request,
        endpoint: EndpointSpec,
    ) -> RequestData:
        has_data = request.method not in ("GET", "DELETE")

        content_type = request.headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()

        use_json = endpoint.json and has_data and media_type == "application/json"
        use_form = endpoint.form and has_data and media_type in self.FORM_MIMETYPE

        req_form = None
        if use_form:
            req_form = get_multidict_items_starlette(
                await request.form(),
                endpoint.form,
            )

        req_json = None
        if use_json:
            req_json = await request.json()
            if req_json is None:
                req_json = {}

        return RequestData(
            query=get_multidict_items_starlette(
                request.query_params,
                endpoint.query,
            ),
            json=req_json,
            form=req_form,
            headers=dict(request.headers),
            cookies=dict(request.cookies),
        )
