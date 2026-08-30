from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RequestData:
    """Normalized request data shared by framework plugins."""

    query: Any = None
    json: Any = None
    form: Any = None
    headers: Any = None
    cookies: Any = None
