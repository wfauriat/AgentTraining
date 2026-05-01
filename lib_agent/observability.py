# observability.py — Phoenix (Arize) plumbing for the agent.
#
# Phoenix uses OpenTelemetry under the hood. We instrument LangChain once
# at module import time; every subsequent ChatOllama / ToolNode / @tool call
# emits OTel spans automatically. No callback handler is passed through
# RunnableConfig — instrumentation works via Python contextvars.
#
# Install + run:
#   pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-langchain
#   docker compose -f observability/docker-compose.phoenix.yml up -d
#   open http://localhost:6006

import os

PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
PROJECT_NAME = os.getenv("PHOENIX_PROJECT", "lib_agent")

_instrumented = False


def setup() -> bool:
    """Configure OTel + instrument LangChain. Idempotent. Returns True on success."""
    global _instrumented
    if _instrumented:
        return True
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        return False

    # register() builds a TracerProvider, attaches an OTLP HTTP exporter
    # pointed at PHOENIX_ENDPOINT, and registers it as the global provider.
    tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint=PHOENIX_ENDPOINT,
        set_global_tracer_provider=True,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    # OpenInference's tracer doesn't implement LangChain 1.x's on_interrupt /
    # on_resume callback hooks, which causes the dispatcher to print errors on
    # every HITL gate. Patch in no-op fallbacks until upstream catches up.
    try:
        from openinference.instrumentation.langchain._tracer import OpenInferenceTracer

        for method in ("on_interrupt", "on_resume"):
            if not hasattr(OpenInferenceTracer, method):
                setattr(OpenInferenceTracer, method, lambda self, *a, **kw: None)
    except Exception:
        pass

    _instrumented = True
    return True


def make_callbacks() -> list:
    """Compatibility shim: Phoenix doesn't need a callback handler.
    Auto-instrumentation lives in OTel context. We just call setup() and
    return [] so chat.py keeps using the same RunnableConfig shape."""
    setup()
    return []


def flush() -> None:
    """Force-flush buffered OTel spans. Call before short-lived exits.

    The base `TracerProvider` API doesn't declare `force_flush`; only the
    SDK subclass does. We resolve via getattr to keep the type checker
    quiet and tolerate a non-SDK provider being installed."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush(timeout_millis=5000)
    except Exception:
        pass
