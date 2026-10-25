"""A fake OpenAI-compatible server for the AI reviewer tests: GET /v1/models and POST /v1/chat/completions on
127.0.0.1, in a thread. No test ever calls a real provider."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

NO_SIGN_ANSWER = '{"boxes": [], "added": [], "no_sign": true, "comment": ""}'


def answer_json(boxes=(), added=(), no_sign=False, comment='') -> str:
    return json.dumps({'boxes': list(boxes), 'added': list(added), 'no_sign': no_sign, 'comment': comment})


class FakeLlm:
    """``answer`` is the assistant text, or a function of the chat body that returns it. ``status`` other than
    200 answers every chat with that HTTP error; ``require_key`` answers 401 without that Bearer key;
    ``redirect`` answers every request with a 302 to that address; ``refuse_json_mode`` answers 400 to a chat
    that asks for response_format, ``refuse_template_kwargs`` to one that sends chat_template_kwargs (OpenAI)."""

    def __init__(self, answer: str | Callable[[dict], str] = NO_SIGN_ANSWER, *, models=('fake-vl',),
                 text_only: bool = False, status: int = 200, delay: float = 0.0, require_key: str | None = None,
                 redirect: str | None = None, refuse_json_mode: bool = False, refuse_template_kwargs: bool = False,
                 error_message: str = 'fake error'):
        self.answer = answer
        self.models = tuple(models)
        self.text_only = text_only
        self.status = status
        self.delay = delay
        self.require_key = require_key
        self.redirect = redirect
        self.refuse_json_mode = refuse_json_mode
        self.refuse_template_kwargs = refuse_template_kwargs
        self.error_message = error_message
        self.requests: list[dict] = []
        self.auth_headers: list[str | None] = []
        self.paths: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _send(self, status: int, data: dict, headers=()) -> None:
                body = json.dumps(data).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                for name, value in headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def _refused(self) -> bool:
                fake.paths.append(self.path)
                fake.auth_headers.append(self.headers.get('Authorization'))
                if fake.redirect is not None:
                    self._send(302, {}, (('Location', fake.redirect),))
                    return True
                if fake.require_key is not None and self.headers.get('Authorization') != 'Bearer ' + fake.require_key:
                    self._send(401, {'error': {'message': fake.error_message}})
                    return True
                return False

            def do_GET(self):
                if self._refused():
                    return
                if self.path != '/v1/models':
                    self._send(404, {'error': {'message': 'not found'}})
                    return
                modalities = ['text'] if fake.text_only else ['text', 'image']
                self._send(200, {'object': 'list', 'data': [
                    {'id': name, 'object': 'model', 'architecture': {'input_modalities': modalities}}
                    for name in fake.models]})

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(length)
                if self._refused():
                    return
                if self.path != '/v1/chat/completions':
                    self._send(404, {'error': {'message': 'not found'}})
                    return
                body = json.loads(raw.decode('utf-8'))
                fake.requests.append(body)
                if fake.refuse_json_mode and 'response_format' in body:
                    self._send(400, {'error': {'message': 'response_format is not supported'}})
                    return
                if fake.refuse_template_kwargs and 'chat_template_kwargs' in body:
                    self._send(400, {'error': {'message': 'Unrecognized request argument: chat_template_kwargs'}})
                    return
                if fake.status != 200:
                    self._send(fake.status, {'error': {'message': fake.error_message}})
                    return
                if fake.delay:
                    time.sleep(fake.delay)
                text = fake.answer(body) if callable(fake.answer) else fake.answer
                self._send(200, {'object': 'chat.completion', 'model': body.get('model'),
                                 'choices': [{'index': 0, 'finish_reason': 'stop',
                                              'message': {'role': 'assistant', 'content': text}}]})

        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._server.daemon_threads = True
        self.base_url = f'http://127.0.0.1:{self._server.server_address[1]}/v1'
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> FakeLlm:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
