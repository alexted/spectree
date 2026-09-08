import asyncio
import inspect
import re
from collections.abc import AsyncIterator, Callable, Mapping
from functools import partial
from typing import Any

try:
    from tempfile import SpooledTemporaryFile
    CachedFile = partial(SpooledTemporaryFile, max_size=1024 * 1024)
except ImportError:
    from io import BytesIO as CachedFile  # type: ignore[assignment]

from falcon import MEDIA_HTML, MEDIA_JSON, Request as FalconRequest, Response as FalconResponse, http_status_to_code
from falcon.asgi import Request as FalconASGIRequest
from falcon.asgi.reader import BufferedReader as ASGIBufferedReader
from falcon.routing.compiled import _FIELD_PATTERN as FALCON_FIELD_PATTERN
from falcon.util.reader import DEFAULT_CHUNK_SIZE, BufferedReader

from spectree.endpoint import EndpointSpec
from spectree.plugins.base import BasePlugin, validate_response
from spectree.request_data import RequestData
from spectree.response import Response


class StreamWrapper:
    def __init__(self, stream: BufferedReader):
        self._buf = CachedFile()
        stream.pipe(self._buf)
        self._buf.seek(0)

    def read(self, size: int | None = -1, /) -> bytes:
        return self._buf.read(size if size is not None else -1)

    def exhaust(self) -> None:
        self._buf.seek(0)
        self._buf.truncate(0)

    def close(self) -> None:
        self._buf.close()

    def __enter__(self) -> "StreamWrapper":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class AsyncStreamWrapper(StreamWrapper):
    def __init__(self):
        self._buf = CachedFile()

    @classmethod
    async def from_stream(cls, stream: ASGIBufferedReader):
        obj = cls()
        loop = asyncio.get_running_loop()
        async for chunk in stream:
            await loop.run_in_executor(None, obj._buf.write, chunk)
        await loop.run_in_executor(None, obj._buf.seek, 0)
        return obj

    async def read(self, size: int | None = -1, /) -> bytes:
        return await asyncio.get_running_loop().run_in_executor(None, super().read, size)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        chunk = await self.read(DEFAULT_CHUNK_SIZE)
        while chunk:
            yield chunk
            chunk = await self.read(DEFAULT_CHUNK_SIZE)

    async def exhaust(self) -> None:
        super().exhaust()

    async def aclose(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.close)

    async def __aenter__(self) -> "AsyncStreamWrapper":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.aclose()


class OpenAPI:
    def __init__(self, spec: Mapping[str, str]):
        self.spec = spec

    def on_get(self, _: Any, resp: Any):
        resp.content_type = MEDIA_JSON
        resp.media = self.spec


class DocPage:
    def __init__(self, html: str, **kwargs: Any):
        self.page = html.format(**kwargs).encode("utf-8")

    def on_get(self, _: Any, resp: Any):
        resp.content_type = MEDIA_HTML
        resp.data = self.page


class OpenAPIAsgi(OpenAPI):
    async def on_get(self, req: Any, resp: Any):
        super().on_get(req, resp)


class DocPageAsgi(DocPage):
    async def on_get(self, req: Any, resp: Any):
        super().on_get(req, resp)


DOC_CLASS = [x.__name__ for x in (DocPage, OpenAPI, DocPageAsgi, OpenAPIAsgi)]
HTTP_500 = "500 Internal Service Response Validation Error"


class FalconPlugin(BasePlugin):
    OPEN_API_ROUTE_CLASS = OpenAPI
    DOC_PAGE_ROUTE_CLASS = DocPage

    def __init__(self, spectree):
        super().__init__(spectree)
        self.ESCAPE = "[" + re.escape(r".()[]?$*+^|") + "]"
        self.ESCAPE_TO = r"\\\g<0>"
        self.EXTRACT = r"{\2}"
        self.INT_ARGS = re.compile(r"((?P<name>\w+)\s*=\s*)?(?P<value>\d+)\s*", re.VERBOSE)
        self.INT_ARGS_NAMES = ("num_digits", "min", "max")

    def register_route(self, app: Any):
        app.add_route(self.config.spec_url, self.OPEN_API_ROUTE_CLASS(self.spectree.spec))
        for ui in self.config.page_templates:
            app.add_route(f"/{self.config.path}/{ui}", self.DOC_PAGE_ROUTE_CLASS(
                self.config.page_templates[ui], spec_url=self.config.filename,
                spec_path=self.config.path, **self.config.swagger_oauth2_config()))

    def find_routes(self):
        routes = []
        def find_node(node):
            if node.resource and node.resource.__class__.__name__ not in DOC_CLASS:
                routes.append(node)
            for child in node.children:
                find_node(child)
        for route in self.spectree.app._router._roots:
            find_node(route)
        return routes

    def parse_func(self, route: Any):
        return route.method_map.items()

    def parse_path(self, route, path_parameter_descriptions):
        subs, parameters = [], []
        for segment in route.uri_template.strip("/").split("/"):
            matches = FALCON_FIELD_PATTERN.finditer(segment)
            if not matches:
                subs.append(segment)
                continue
            escaped = re.sub(self.ESCAPE, self.ESCAPE_TO, segment)
            subs.append(FALCON_FIELD_PATTERN.sub(self.EXTRACT, escaped))
            for field in matches:
                variable, converter, argstr = [field.group(name) for name in ("fname", "cname", "argstr")]
                if converter == "int":
                    arg_values = [None, None, None]
                    for i, match in enumerate(self.INT_ARGS.finditer(argstr or "")):
                        name, value = match.group("name"), match.group("value")
                        index = self.INT_ARGS_NAMES.index(name) if name else i
                        arg_values[index] = value
                    num_digits, minimum, maximum = arg_values
                    schema = {"type": "integer", "format": f"int{num_digits}" if num_digits else "int32"}
                    if minimum:
                        schema["minimum"] = minimum
                    if maximum:
                        schema["maximum"] = maximum
                elif converter == "uuid":
                    schema = {"type": "string", "format": "uuid"}
                elif converter == "dt":
                    schema = {"type": "string", "format": "date-time"}
                else:
                    schema = {"type": "string"}
                description = path_parameter_descriptions.get(variable, "") if path_parameter_descriptions else ""
                parameters.append({"name": variable, "in": "path", "required": True, "schema": schema, "description": description})
        return f"/{'/'.join(subs)}", parameters

    def get_request_data(self, req: FalconRequest, endpoint: EndpointSpec) -> RequestData:
        req_form = None
        if endpoint.form and req.content_type:
            req_form = {}
            if req.content_type == "application/x-www-form-urlencoded":
                req_form = req.get_media()
            elif req.content_type.startswith("multipart/form-data"):
                for part in req.get_media():
                    if part.filename is None:
                        req_form[part.name] = part.get_data()
                    else:
                        req_form[part.name] = part
                        part.stream = StreamWrapper(part.stream)
        return RequestData(
            query=dict(req.params),
            json=req.get_media(default_when_empty={}) if endpoint.json else None,
            form=req_form,
            headers=req.headers,
            cookies=req.cookies,
        )

    def validate_response(self, resp: FalconResponse, resp_model: Response | None, skip_validation: bool, force_resp_serialize: bool):
        resp_validation_error = None
        if not self._data_set_manually(resp):
            if not skip_validation and resp_model:
                try:
                    status = http_status_to_code(resp.status)
                    result = validate_response(self.model_adapter, resp_model.find_model(status), resp.media, force_resp_serialize)
                except self.model_adapter.validation_error as err:
                    resp_validation_error = err
                    resp.status = HTTP_500
                    resp.media = self.model_adapter.validation_errors(err)
                else:
                    if isinstance(result.payload, bytes):
                        resp.data = result.payload
                        resp.content_type = MEDIA_JSON
                    else:
                        resp.media = result.payload
            elif self.model_adapter.is_partial_model_instance(resp.media):
                resp.data = self.model_adapter.dump_json(resp.media)
                resp.content_type = MEDIA_JSON
        return resp_validation_error

    def validate(self, func: Callable, endpoint: EndpointSpec, *args: Any, **kwargs: Any):
        _self, req, resp = args[:3]
        request_data = self.get_request_data(req, endpoint)
        req_validation_error = None
        if not endpoint.skip_validation:
            try:
                request_data = self.validate_request_data(request_data, endpoint)
            except self.model_adapter.validation_error as err:
                req_validation_error = err
                resp.status = f"{endpoint.validation_error_status} Validation Error"
                resp.media = self.model_adapter.validation_errors(err)
        self.set_request_data(req, request_data)
        endpoint.before(req, resp, req_validation_error, _self, self.model_adapter)
        if req_validation_error:
            return None
        self.inject_request_data(request_data, endpoint, kwargs)
        result = func(*args, **kwargs)
        resp_validation_error = self.validate_response(resp, endpoint.response, endpoint.skip_validation, endpoint.force_resp_serialize)
        endpoint.after(req, resp, resp_validation_error, _self, self.model_adapter)
        return result

    @staticmethod
    def _data_set_manually(resp):
        return (resp.text is not None or resp.data is not None) and resp.media is None

    def bypass(self, func, method):
        return True if isinstance(func, partial) else inspect.isfunction(func)


class FalconAsgiPlugin(FalconPlugin):
    ASYNC = True
    OPEN_API_ROUTE_CLASS = OpenAPIAsgi
    DOC_PAGE_ROUTE_CLASS = DocPageAsgi

    async def get_request_data(self, req: FalconASGIRequest, endpoint: EndpointSpec) -> RequestData:
        req_form = None
        if endpoint.form and req.content_type:
            req_form = {}
            if req.content_type == "application/x-www-form-urlencoded":
                req_form = await req.get_media()
            elif req.content_type.startswith("multipart/form-data"):
                async for part in await req.get_media():
                    if part.filename is None:
                        req_form[part.name] = await part.get_data()
                    else:
                        req_form[part.name] = part
                        part.stream = await AsyncStreamWrapper.from_stream(part.stream)
        return RequestData(
            query=dict(req.params),
            json=await req.get_media(default_when_empty={}) if endpoint.json else None,
            form=req_form,
            headers=req.headers,
            cookies=req.cookies,
        )

    async def validate(self, func: Callable, endpoint: EndpointSpec, *args: Any, **kwargs: Any):
        _self, req, resp = args[:3]
        request_data = await self.get_request_data(req, endpoint)
        req_validation_error = None
        if not endpoint.skip_validation:
            try:
                request_data = self.validate_request_data(request_data, endpoint)
            except self.model_adapter.validation_error as err:
                req_validation_error = err
                resp.status = f"{endpoint.validation_error_status} Validation Error"
                resp.media = self.model_adapter.validation_errors(err)
        self.set_request_data(req, request_data)
        endpoint.before(req, resp, req_validation_error, _self, self.model_adapter)
        if req_validation_error:
            return None
        self.inject_request_data(request_data, endpoint, kwargs)
        result = await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)
        resp_validation_error = self.validate_response(resp, endpoint.response, endpoint.skip_validation, endpoint.force_resp_serialize)
        endpoint.after(req, resp, resp_validation_error, _self, self.model_adapter)
        return result
