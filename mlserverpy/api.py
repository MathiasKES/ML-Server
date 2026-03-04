from __future__ import annotations
from typing import Any, Optional
from .client import Client
from .exceptions import NotInitializedError

_client: Optional[Client] = None

def init(host: str, username: str | None = None, password: str | None = None, token: str | None = None):
    global _client
    _client = Client(host=host, username=username, password=password, token=token)
    return _client

def _c() -> Client:
    if _client is None:
        raise NotInitializedError("mlserverpy.init(...) must be called first.")
    return _client

def run(**kwargs) -> str:
    return _c().run(**kwargs)

def log_metric(**kwargs):
    _c().log_metric(**kwargs)

def log_scalar(**kwargs):
    _c().log_scalar(**kwargs)

def post(**kwargs):
    return _c().post(**kwargs)

def flush(**kwargs):
    _c().flush(**kwargs)

def sync(**kwargs):
    _c().sync(**kwargs)

def heartbeat(**kwargs):
    _c().heartbeat(**kwargs)

def get(*args: Any, **kwargs: Any):
    raise NotImplementedError("Retrieval API not implemented yet.")
