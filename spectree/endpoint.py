from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from spectree._types import HookHandler
from spectree.model_adapter import ModelSpec
from spectree.response import Response


REQUEST_MODEL_ARGUMENTS = (
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

    Annotation-derived injection is represented by ``injected_arguments``.
    The runtime plugin must not inspect the original function annotations again.
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
        return getattr(self, name)