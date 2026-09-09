from types import SimpleNamespace

import pytest

from spectree.endpoint import EndpointSpec
from spectree.plugins.base import BasePlugin, Context
from spectree.request_data import RequestData


class StubAdapter:
    validation_error = ValueError

    def __init__(self):
        self.calls = []

    def validate_obj(self, model, value):
        self.calls.append((model, value))
        return model, value


def make_endpoint(**overrides):
    values = dict(
        query=None,
        json=None,
        form=None,
        headers=None,
        cookies=None,
        request_model_keys=(),
        response=None,
        injected_arguments=frozenset(),
        before=lambda *args: None,
        after=lambda *args: None,
        validation_error_status=422,
        skip_validation=False,
        force_resp_serialize=False,
        tags=(),
        security=None,
        deprecated=False,
        path_parameter_descriptions=None,
        operation_id=None,
    )
    values.update(overrides)
    return EndpointSpec(**values)


def make_plugin():
    plugin = BasePlugin.__new__(BasePlugin)
    plugin.model_adapter = StubAdapter()
    return plugin


def test_request_data_is_frozen_and_slotted():
    request_data = RequestData(json={"name": "alice"})

    with pytest.raises(AttributeError):
        request_data.json = {"name": "bob"}

    with pytest.raises(AttributeError):
        request_data.extra = "value"


def test_request_data_validation_covers_all_request_fields():
    plugin = make_plugin()

    models = {
        "query": object(),
        "json": object(),
        "form": object(),
        "headers": object(),
        "cookies": object(),
    }

    raw = RequestData(
        query={"q": "1"},
        json={"name": "alice"},
        form={"field": "value"},
        headers={"X-Test": "yes"},
        cookies={"session": "abc"},
    )

    endpoint = make_endpoint(
        query=models["query"],
        json=models["json"],
        form=models["form"],
        headers=models["headers"],
        cookies=models["cookies"],
    )

    result = plugin.validate_request_data(raw, endpoint)

    assert result == RequestData(
        query=(models["query"], raw.query),
        json=(models["json"], raw.json),
        form=(models["form"], raw.form),
        headers=(models["headers"], raw.headers),
        cookies=(models["cookies"], raw.cookies),
    )

    assert plugin.model_adapter.calls == [
        (models["query"], raw.query),
        (models["json"], raw.json),
        (models["form"], raw.form),
        (models["headers"], raw.headers),
        (models["cookies"], raw.cookies),
    ]


def test_request_data_validation_drops_unmodeled_values():
    plugin = make_plugin()

    raw = RequestData(
        query={"q": "1"},
        json={"name": "alice"},
        form={"field": "value"},
        headers={"X-Test": "yes"},
        cookies={"session": "abc"},
    )

    endpoint = make_endpoint(
        query=object(),
    )

    result = plugin.validate_request_data(raw, endpoint)

    assert result.query[1] == raw.query
    assert result.json is None
    assert result.form is None
    assert result.headers is None
    assert result.cookies is None


def test_request_data_validation_does_not_call_adapter_for_none():
    plugin = make_plugin()

    model = object()
    endpoint = make_endpoint(json=model)

    result = plugin.validate_request_data(
        RequestData(json=None),
        endpoint,
    )

    assert result.json is None
    assert plugin.model_adapter.calls == []


def test_request_data_injection_uses_only_declared_arguments():
    plugin = make_plugin()

    request_data = RequestData(
        query={"q": "1"},
        json={"name": "alice"},
        form={"field": "value"},
        headers={"X-Test": "yes"},
        cookies={"session": "abc"},
    )

    endpoint = make_endpoint(
        injected_arguments=frozenset({"query", "json"}),
    )

    kwargs = {"existing": True}

    plugin.inject_request_data(
        request_data,
        endpoint,
        kwargs,
    )

    assert kwargs == {
        "existing": True,
        "query": {"q": "1"},
        "json": {"name": "alice"},
    }


def test_set_request_data_replaces_missing_context():
    request = SimpleNamespace(context=None)
    request_data = RequestData(json={"name": "alice"})

    BasePlugin.set_request_data(request, request_data)

    assert request.context is request_data


def test_set_request_data_replaces_existing_request_data_context():
    request = SimpleNamespace(
        context=RequestData(json={"old": True}),
    )
    request_data = RequestData(json={"new": True})

    BasePlugin.set_request_data(request, request_data)

    assert request.context is request_data


def test_set_request_data_replaces_legacy_context():
    request = SimpleNamespace(
        context=Context(
            query="old-query",
            json="old-json",
            form="old-form",
            headers="old-headers",
            cookies="old-cookies",
        )
    )
    request_data = RequestData(json={"new": True})

    BasePlugin.set_request_data(request, request_data)

    assert request.context is request_data


def test_set_request_data_updates_mapping_context():
    context = {"keep": "value"}
    request = SimpleNamespace(context=context)
    request_data = RequestData(
        query={"q": "1"},
        json={"name": "alice"},
        form={"field": "value"},
        headers={"X-Test": "yes"},
        cookies={"session": "abc"},
    )

    BasePlugin.set_request_data(request, request_data)

    assert request.context is context
    assert context == {
        "keep": "value",
        "query": request_data.query,
        "json": request_data.json,
        "form": request_data.form,
        "headers": request_data.headers,
        "cookies": request_data.cookies,
    }


def test_set_request_data_preserves_framework_context_object():
    context = SimpleNamespace(
        keep="value",
        query="old-query",
        json="old-json",
        form="old-form",
        headers="old-headers",
        cookies="old-cookies",
    )
    request = SimpleNamespace(context=context)

    request_data = RequestData(
        query={"q": "1"},
        json=None,
        form={"field": "value"},
        headers={"X-Test": "yes"},
        cookies={"session": "abc"},
    )

    BasePlugin.set_request_data(request, request_data)

    assert request.context is context
    assert context.keep == "value"
    assert context.query == request_data.query
    assert context.json is None
    assert context.form == request_data.form
    assert context.headers == request_data.headers
    assert context.cookies == request_data.cookies


def test_request_data_is_not_mutated_by_set_request_data():
    request = SimpleNamespace(context=None)
    request_data = RequestData(
        query={"q": "1"},
        json={"name": "alice"},
    )

    BasePlugin.set_request_data(request, request_data)

    assert request.context is request_data
    assert request_data == RequestData(
        query={"q": "1"},
        json={"name": "alice"},
    )


def test_base_plugin_exposes_request_extraction_boundary():
    plugin = make_plugin()
    endpoint = make_endpoint()

    with pytest.raises(NotImplementedError):
        plugin.get_request_data(SimpleNamespace(), endpoint)


def test_set_request_data_replaces_legacy_context():
    request = SimpleNamespace(
        context=Context(
            query="old-query",
            json="old-json",
            form="old-form",
            headers="old-headers",
            cookies="old-cookies",
        )
    )
    request_data = RequestData(json={"new": True})

    BasePlugin.set_request_data(request, request_data)

    assert request.context is request_data
