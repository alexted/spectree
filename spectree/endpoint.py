from dataclasses import dataclass
from typing import Any, Mapping

from spectree._types import HookHandler
from spectree.model_adapter import ModelSpec
from spectree.response import Response

REQUEST_MODEL_ARGUMENTS: tuple[str, ...] = (
    "query",
    "json",
    "form",
    "headers",
    "cookies",
)


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """
    Immutable runtime contract compiled from a SpecTree endpoint declaration.

    Request-model annotations and OpenAPI component names are resolved once
    during decoration. Runtime framework plugins consume this contract directly
    and must not inspect endpoint annotations or legacy function attributes.
    """

    query: ModelSpec | None
    json: ModelSpec | None
    form: ModelSpec | None
    headers: ModelSpec | None
    cookies: ModelSpec | None

    request_model_keys: tuple[tuple[str, str], ...]

    response: Response | None
    injected_arguments: frozenset[str]

    before: HookHandler
    after: HookHandler

    validation_error_status: int
    skip_validation: bool
    force_resp_serialize: bool

    tags: tuple[Any, ...]
    security: Any
    deprecated: bool
    path_parameter_descriptions: Mapping[str, str] | None
    operation_id: str | None

    def model_for(self, name: str) -> ModelSpec | None:
        if name not in REQUEST_MODEL_ARGUMENTS:
            raise KeyError(name)
        return getattr(self, name)

    def model_key_for(self, name: str) -> str | None:
        for model_name, model_key in self.request_model_keys:
            if model_name == name:
                return model_key
        return None
