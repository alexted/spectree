from types import SimpleNamespace

import pytest

from spectree.endpoint import EndpointSpec
from spectree.plugins.base import BasePlugin
from spectree.request_data import RequestData


class StubAdapter:
    def validate_obj(self, model, value):
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


def test_request_data_is_immutable_and_slotted():
    data = RequestData(json={"name": "alice"})

    with pytest.raises(AttributeError):
        data.json = {}

    with pytest.raises(AttributeError):
        data.extra = "value"


def test_validate_request_data_validates_all_declared_models():
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
    endpoint = make_endpoint(**models)

    result = plugin.validate_request_data(raw, endpoint)

    assert result == RequestData(
        query=(models["query"], raw.query),
        json=(models["json"], raw.json),
        form=(models["form"], raw.form),
        headers=(models["headers"], raw.headers),
        cookies=(models["cookies"], raw.cookies),
    )


def test_validate_request_data_preserves_unmodeled_fields():
    plugin = make_plugin()
    raw = RequestData(
        query={"q": "1"},
        json={"name": "alice"},
        form={"field": "value"},
        headers={"X-Test": "yes"},
        cookies={"session": "abc"},
    )
    endpoint = make_endpoint(json=object())

    result = plugin.validate_request_data(raw, endpoint)

    assert result.query == raw.query
    assert result.json == (endpoint.json, raw.json)
    assert result.form == raw.form
    assert result.headers == raw.headers
    assert result.cookies == raw.cookies


def test_inject_request_data_uses_only_declared_arguments():
    plugin = make_plugin()
    data = RequestData(
        query={"q": "1"},
        json={"name": "alice"},
        form={"field": "value"},
        headers={"X-Test": "yes"},
        cookies={"session": "abc"},
    )
    endpoint = make_endpoint(injected_arguments=frozenset({"query", "json"}))
    kwargs = {"existing": True}

    plugin.inject_request_data(data, endpoint, kwargs)

    assert kwargs == {"existing": True, "query": data.query, "json": data.json}


def test_set_request_data_uses_request_data_as_context_when_missing():
    request = SimpleNamespace(context=None)
    data = RequestData(json={"name": "alice"})

    BasePlugin.set_request_data(request, data)

    assert request.context is data


def test_set_request_data_replaces_previous_request_data_context():
    request = SimpleNamespace(context=RequestData(json={"old": True}))
    data = RequestData(json={"new": True})

    BasePlugin.set_request_data(request, data)

    assert request.context is data


def test_set_request_data_preserves_application_context_attributes():
    context = SimpleNamespace(application_value="keep")
    request = SimpleNamespace(context=context)
    data = RequestData(json={"name": "alice"})

    BasePlugin.set_request_data(request, data)

    assert request.context is context
    assert context.application_value == "keep"
    assert context.query is None
    assert context.json == data.json
    assert context.form is None
    assert context.headers is None
    assert context.cookies is None


def test_set_request_data_supports_mapping_contexts():
    context = {}
    request = SimpleNamespace(context=context)
    data = RequestData(json={"name": "alice"})

    BasePlugin.set_request_data(request, data)

    assert context == {
        "query": None,
        "json": data.json,
        "form": None,
        "headers": None,
        "cookies": None,
    }


def test_get_request_data_is_an_explicit_plugin_boundary():
    plugin = make_plugin()

    with pytest.raises(NotImplementedError):
        plugin.get_request_data(SimpleNamespace(), make_endpoint())
