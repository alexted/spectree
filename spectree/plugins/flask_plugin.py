from collections.abc import Callable
from typing import Any

import flask
from flask import Blueprint, abort, current_app, jsonify, make_response, request

from spectree.endpoint import EndpointSpec
from spectree.plugins.base import validate_response
from spectree.plugins.werkzeug_utils import WerkzeugPlugin, flask_response_unpack
from spectree.request_data import RequestData
from spectree.response import Response
from spectree.utils import get_multidict_items


class FlaskPlugin(WerkzeugPlugin):
    def get_current_app(self):
        return current_app

    def is_app_response(self, resp):
        return isinstance(resp, flask.Response)

    @staticmethod
    def is_blueprint(app: Any) -> bool:
        return isinstance(app, Blueprint)

    def get_request_data(self, request, endpoint: EndpointSpec) -> RequestData:
        has_data = request.method not in ("GET", "DELETE")
        use_json = (
            endpoint.json and has_data and request.mimetype not in self.FORM_MIMETYPE
        )
        use_form = endpoint.form and has_data and request.mimetype in self.FORM_MIMETYPE
        return RequestData(
            query=get_multidict_items(request.args, endpoint.query),
            json=(request.get_json(silent=True) or {}) if use_json else None,
            form=self.fill_form(request) if use_form else None,
            headers=dict(request.headers),
            cookies=get_multidict_items(request.cookies),
        )

    def validate_response(
        self,
        resp,
        resp_model: Response | None,
        skip_validation: bool,
        force_resp_serialize: bool,
    ):
        resp_validation_error = None
        payload, status, additional_headers = flask_response_unpack(resp)

        if self.is_app_response(payload):
            resp_status, resp_headers = payload.status_code, payload.headers
            payload = payload.get_data()
            if status == 200:
                status = resp_status
            resp_headers.extend(additional_headers)
            additional_headers = resp_headers

        if not skip_validation and resp_model:
            try:
                result = validate_response(
                    self.model_adapter,
                    resp_model.find_model(status),
                    payload,
                    force_resp_serialize,
                )
            except self.model_adapter.validation_error as err:
                response = make_response(self.model_adapter.validation_errors(err), 500)
                resp_validation_error = err
            else:
                response = make_response(
                    self.get_current_app().response_class(
                        result.payload,
                        mimetype="application/json",
                    )
                    if isinstance(result.payload, bytes)
                    else result.payload,
                    status,
                    additional_headers,
                )
        else:
            if self.model_adapter.is_partial_model_instance(payload):
                payload = self.get_current_app().response_class(
                    self.model_adapter.dump_json(payload),
                    mimetype="application/json",
                )
            response = make_response(payload, status, additional_headers)
        return response, resp_validation_error

    def validate(
        self, func: Callable, endpoint: EndpointSpec, *args: Any, **kwargs: Any
    ):
        if request.url_rule is not None and getattr(
            request.url_rule, "websocket", False
        ):
            return func(*args, **kwargs)

        request_data = self.get_request_data(request, endpoint)
        response = None
        req_validation_error = None
        try:
            if not endpoint.skip_validation:
                request_data = self.validate_request_data(request_data, endpoint)
            self.set_request_data(request, request_data)
        except self.model_adapter.validation_error as err:
            req_validation_error = err
            response = make_response(
                jsonify(self.model_adapter.validation_errors(err)),
                endpoint.validation_error_status,
            )

        endpoint.before(
            request,
            response,
            req_validation_error,
            None,
            self.model_adapter,
        )
        if req_validation_error is not None:
            assert response
            abort(response)

        self.inject_request_data(request_data, endpoint, kwargs)
        result = func(*args, **kwargs)
        response, resp_validation_error = self.validate_response(
            result,
            endpoint.response,
            endpoint.skip_validation,
            endpoint.force_resp_serialize,
        )
        endpoint.after(
            request,
            response,
            resp_validation_error,
            None,
            self.model_adapter,
        )
        return response
