from types import SimpleNamespace

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


def test_request_data_is_immutable():
    data = RequestData(json={"name": "alice"})

    try:
        data.json = {}
    except AttributeError:
        pass
    else:
        raise AssertionError("RequestData must be immutable")


def test_request_data_is_validated_and_injected():
    plugin = make_plugin()
    model = object()
    endpoint = make_endpoint(
        json=model,
        injected_arguments=frozenset({"json"}),
    )

    validated = plugin.validate_request_data(
        RequestData(json={"name": "alice"}),
        endpoint,
    )

    assert validated.json == (model, {"name": "alice"})

    kwargs = {}
    plugin.inject_request_data(validated, endpoint, kwargs)
    assert kwargs == {"json": (model, {"name": "alice"})}


def test_set_request_data_preserves_framework_context():
    plugin = make_plugin()
    request = SimpleNamespace(context=SimpleNamespace(application_value="keep"))
    data = RequestData(json={"name": "alice"})

    plugin.set_request_data(request, data)

    assert request.context.application_value == "keep"
    assert request.context.json == {"name": "alice"}


def test_set_request_data_replaces_previous_request_data_context():
    plugin = make_plugin()
    request = SimpleNamespace(context=RequestData(json={"old": True}))
    data = RequestData(json={"new": True})

    plugin.set_request_data(request, data)

    assert request.context is data