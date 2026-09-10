import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NamedTuple,
    Optional,
    TypeVar,
)
from spectree._types import JsonType, ModelAdapterType
from spectree.config import Configuration
from spectree.request_data import RequestData
from spectree.endpoint import EndpointSpec
from spectree.model_adapter import ModelSpec

if TYPE_CHECKING:
    from spectree.spec import SpecTree


class Context(NamedTuple):
    query: Any | None
    json: Any | None
    form: Any | None
    headers: Any | None
    cookies: Any | None


BackendRoute = TypeVar("BackendRoute")


class BasePlugin(Generic[BackendRoute]):
    """Base plugin for SpecTree plugin classes."""

    ASYNC = False
    FORM_MIMETYPE = ("application/x-www-form-urlencoded", "multipart/form-data")

    def __init__(self, spectree: "SpecTree"):
        self.spectree = spectree
        self.config: Configuration = spectree.config
        self.model_adapter: ModelAdapterType = spectree.model_adapter
        self.logger = logging.getLogger(__name__)

    def register_route(self, app: Any):
        raise NotImplementedError

    def validate(
        self, func: Callable, endpoint: EndpointSpec, *args: Any, **kwargs: Any
    ):
        raise NotImplementedError

    def get_request_data(
        self, request: Any, endpoint: EndpointSpec
    ) -> RequestData | Awaitable[RequestData]:
        raise NotImplementedError

    def validate_request_data(
        self,
        func: Callable,
        endpoint: EndpointSpec,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Validate request/response and invoke the endpoint using
        the precomputed endpoint contract.
        """
        raise NotImplementedError

    def find_routes(self) -> BackendRoute:
        raise NotImplementedError

    def bypass(self, func: Callable, method: str) -> bool:
        raise NotImplementedError

    def parse_path(
        self, route: Any, path_parameter_descriptions: Mapping[str, str] | None
    ):
        raise NotImplementedError

    def parse_func(self, route: BackendRoute):
        raise NotImplementedError

    def get_func_operation_id(self, func: Callable, path: str, method: str):
        operation_id = getattr(func, "operation_id", None)
        if not operation_id:
            operation_id = f"{method.lower()}_{path.replace('/', '_')}"
        return operation_id


@dataclass(frozen=True)
class RawResponsePayload:
    payload: JsonType | bytes


@dataclass(frozen=True)
class ResponseValidationResult:
    payload: Any


def validate_response(
    model_adapter: ModelAdapterType,
    validation_model: Optional[ModelSpec],
    response_payload: Any,
    force_serialize: bool = False,
) -> ResponseValidationResult:
    """Validate a given ``response_payload`` against a ``validation_model``.
    This does nothing if ``validation_model is None``.

    :param validation_model: model class used to validate the provided
        ``response_payload``.
    :param response_payload: Validated response payload. A :class:`RawResponsePayload`
        should be provided when the plugin view function returned an already
        JSON-serialized response payload.
    :param force_serialize: Always serialize the validation model instance.
    """
    if validation_model is None:
        return ResponseValidationResult(payload=response_payload)

    final_response_payload: Any
    skip_validation = False
    if isinstance(response_payload, RawResponsePayload):
        final_response_payload = response_payload.payload
    elif model_adapter.is_model_instance(response_payload, validation_model):
        skip_validation = True
        final_response_payload = model_adapter.dump_json(response_payload)
    else:
        final_response_payload = response_payload

    if not skip_validation:
        if isinstance(final_response_payload, bytes):
            validated_instance = model_adapter.validate_json(
                validation_model, final_response_payload
            )
        else:
            validated_instance = model_adapter.validate_obj(
                validation_model, final_response_payload
            )
        if force_serialize or model_adapter.is_partial_model_instance(
            final_response_payload
        ):
            final_response_payload = model_adapter.dump_json(validated_instance)

    return ResponseValidationResult(payload=final_response_payload)
