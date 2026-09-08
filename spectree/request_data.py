from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RequestData:
    """Framework-independent request data passed through Spectree's runtime."""

    query: Any = None
    json: Any = None
    form: Any = None
    headers: Any = None
    cookies: Any = None
