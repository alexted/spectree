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

    Request-model annotations are resolved once during decoration and stored in
    ``injected_arguments``. Runtime framework plugins consume this contract
    directly and must not inspect endpoint annotations.
    """

    query: ModelSpec | None
    json: ModelSpec | None
    form: ModelSpec | None
    headers: ModelSpec | None
    cookies: ModelSpec | None

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