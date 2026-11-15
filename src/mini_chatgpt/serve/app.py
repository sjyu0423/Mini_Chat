"""Local HTTP API for the chat demo (M6).

Endpoints
---------
* ``GET  /api/health`` — ``status`` is ``"loading"``, ``"ready"`` or ``"error"``; when
  ready, also model info, generation defaults, request limits and MPS memory.
* ``POST /api/chat`` — body ``{"message": str, "history": [{"role", "content"}...],
  "settings": {"temperature", "max_new_tokens", "seed"}}`` (history and settings
  optional). Returns the reply text, its ``stop_reason`` (``end_of_turn``,
  ``max_tokens``, ``structural_token``, ``context_full``), token counts and timing.
  Validation errors are 422, an over-long prompt 413, a busy/not-ready model 503, and a
  failure inside generation 500 with ``stop_reason: "error"``.

If a built frontend exists (``apps/chat-web/dist``), it is served at ``/`` so one
process hosts the whole demo. The server is meant to bind to 127.0.0.1 only.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from mini_chatgpt.chat.format import Message
from mini_chatgpt.serve.engine import ChatEngine, EngineBusyError, Limits

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float | None = None
    max_new_tokens: int | None = None
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    settings: GenerationSettings | None = None


class ChatResponse(BaseModel):
    reply: str
    stop_reason: str
    new_tokens: int
    prompt_tokens: int
    dropped_history_messages: int
    duration_ms: float
    tokens_per_s: float
    seed: int
    settings: dict[str, Any]


def validate_request(req: ChatRequest, limits: Limits) -> list[Message]:
    """Return the full message list or raise ``HTTPException(422)`` with a reason."""

    def bad(msg: str) -> HTTPException:
        return HTTPException(status_code=422, detail=msg)

    if not req.message.strip():
        raise bad("message must not be empty")
    if len(req.message) > limits.max_message_chars:
        raise bad(f"message longer than {limits.max_message_chars} characters")
    if len(req.history) > limits.max_history_messages:
        raise bad(f"history longer than {limits.max_history_messages} messages")
    if len(req.history) % 2:
        raise bad("history must contain complete user/assistant pairs")
    for i, m in enumerate(req.history):
        expected = "user" if i % 2 == 0 else "assistant"
        if m.role != expected:
            raise bad(f"history[{i}] must have role '{expected}'")
        if not m.content.strip():
            raise bad(f"history[{i}] is empty")
        if len(m.content) > limits.max_message_chars:
            raise bad(f"history[{i}] longer than {limits.max_message_chars} characters")
    total = len(req.message) + sum(len(m.content) for m in req.history)
    if total > limits.max_total_chars:
        raise bad(f"conversation longer than {limits.max_total_chars} characters")
    s = req.settings
    if s is not None:
        if s.temperature is not None and not (
            limits.min_temperature <= s.temperature <= limits.max_temperature
        ):
            raise bad(
                f"temperature must be between {limits.min_temperature} and {limits.max_temperature}"
            )
        if s.max_new_tokens is not None and not (
            limits.min_new_tokens <= s.max_new_tokens <= limits.max_new_tokens
        ):
            raise bad(
                f"max_new_tokens must be between {limits.min_new_tokens} and "
                f"{limits.max_new_tokens}"
            )
    messages: list[Message] = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})
    return messages


class EngineHolder:
    """Loads the engine in a background thread so ``/api/health`` can report progress."""

    def __init__(self, factory: Callable[[], ChatEngine]) -> None:
        self.factory = factory
        self.engine: ChatEngine | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        def load() -> None:
            try:
                self.engine = self.factory()
                logger.info("model ready: %s", self.engine.info)
            except Exception as exc:  # reported through /api/health
                logger.exception("model failed to load")
                self.error = f"{type(exc).__name__}: {exc}"

        self.thread = threading.Thread(target=load, name="model-loader", daemon=True)
        self.thread.start()

    def wait(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)


def create_app(factory: Callable[[], ChatEngine], *, static_dir: Path | None = None) -> FastAPI:
    holder = EngineHolder(factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        holder.start()
        yield

    app = FastAPI(title="mini_chatgpt local demo", version="1.0", lifespan=lifespan)
    app.state.holder = holder

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        if holder.error:
            return {"status": "error", "error": holder.error}
        if holder.engine is None:
            return {"status": "loading"}
        return {"status": "ready", **holder.engine.describe()}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> Any:
        engine = holder.engine
        if engine is None:
            raise HTTPException(status_code=503, detail=holder.error or "model is loading")
        messages = validate_request(req, engine.limits)
        s = req.settings or GenerationSettings()
        try:
            result = engine.generate(
                messages, temperature=s.temperature, max_new_tokens=s.max_new_tokens, seed=s.seed
            )
        except EngineBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:  # prompt does not fit the context window
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("generation failed")
            return JSONResponse(
                status_code=500,
                content={"stop_reason": "error", "detail": f"{type(exc).__name__}: {exc}"},
            )
        return ChatResponse(**result.__dict__)

    if static_dir is not None and (static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app
