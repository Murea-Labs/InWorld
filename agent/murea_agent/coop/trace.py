"""Full text diagnostics for co-op operations; no credentials or binary assets."""

from contextlib import contextmanager
import time
from murea_agent.providers import llm
from murea_agent import reqlog


@contextmanager
def operation(world_id, round_number, request_id, stage="agent"):
    token = reqlog.bind(world_id)
    try:
        with reqlog.scope(round=round_number, request_id=request_id, stage=stage, full=True):
            yield
    finally:
        reqlog.unbind(token)


def chat(**kwargs):
    start = time.monotonic()
    tag = kwargs.get("tag", "coop")
    reqlog.record(
        "agent_request",
        tag=tag,
        model=kwargs.get("model"),
        request={
            k: v
            for k, v in kwargs.items()
            if k in ("system", "user", "history", "schema", "temperature", "thinking")
        },
    )
    try:
        result = llm.chat(**kwargs)
        reqlog.record("agent_result", tag=tag, ok=True, response=result, seconds=time.monotonic() - start)
        return result
    except Exception as exc:
        reqlog.record("agent_result", tag=tag, ok=False, error=str(exc), seconds=time.monotonic() - start)
        raise
