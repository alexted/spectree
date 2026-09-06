from collections.abc import Callable, Iterator
from typing import Any, Protocol

from spectree.model_adapter.protocol import ModelAdapter, ModelSpec


NamingStrategy = Callable[[ModelSpec], str]
NestedNamingStrategy = Callable[[str, str], str]
ModelAdapterType = ModelAdapter[Any, Exception, Any]

HookHandler = Callable[
    [Any, Any, Exception | None, Any, ModelAdapterType],
    Any,
]


class MultiDict(Protocol):
    def get(self, key: str) -> str | None:
        ...

    def getlist(self, key: str) -> list[str]:
        ...

    def __iter__(self) -> Iterator[str]:
        ...


class MultiDictStarlette(Protocol):
    def __iter__(self) -> Iterator[str]:
        ...

    def getlist(self, key: Any) -> list[Any]:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...


JsonType = int | str | bool | list["JsonType"] | dict[str, "JsonType"] | None
