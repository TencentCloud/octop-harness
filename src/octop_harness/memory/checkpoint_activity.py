"""Keep PostgreSQL retention outside a complete graph read/execute/write cycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from contextvars import copy_context
from typing import Any


async def _settle(task: asyncio.Task[Any]) -> Any:
    """Join a worker even if its async owner receives repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


@asynccontextmanager
async def checkpoint_activity(checkpointer: Any, thread_id: str) -> AsyncIterator[None]:
    """Hold a supported saver's activity lease without blocking the event loop.

    Older savers remain compatible. Hosts using a raw graph must guard the
    complete operation, including state reads and the
    final checkpoint flush. Worker acquisition/release is joined on cancellation
    so an abandoned request cannot leak a PostgreSQL advisory lock.
    """
    if not thread_id:
        yield
        return
    if callable(getattr(type(checkpointer), "acheckpoint_activity", None)):
        # Memory's async scope also tracks accepted saver I/O and rejects late
        # children after graph cancellation; the sync lease alone cannot do so.
        async with checkpointer.acheckpoint_activity(thread_id):
            yield
        return
    factory = getattr(type(checkpointer), "checkpoint_activity", None)
    if not callable(factory):
        yield
        return
    context = checkpointer.checkpoint_activity(thread_id)
    acquire = asyncio.create_task(asyncio.to_thread(context.__enter__))
    try:
        await asyncio.shield(acquire)
    except asyncio.CancelledError:
        await _settle(acquire)
        await _settle(asyncio.create_task(asyncio.to_thread(context.__exit__, None, None, None)))
        raise
    try:
        yield
    finally:
        release = asyncio.create_task(asyncio.to_thread(context.__exit__, None, None, None))
        try:
            await asyncio.shield(release)
        except asyncio.CancelledError:
            await _settle(release)
            raise


async def _stream_with_activity(
    source: AsyncIterator[Any], checkpointer: Any, thread_id: str
) -> AsyncGenerator[Any, None]:
    """Guard a pull-driven stream until exhaustion, error, cancellation or close."""
    async with checkpoint_activity(checkpointer, thread_id):
        try:
            async for item in source:
                yield item
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()


async def _advance(source: AsyncGenerator[Any, None]) -> Any:
    return await anext(source)


async def _close_stream(source: AsyncGenerator[Any, None]) -> None:
    await source.aclose()


async def _guarded_checkpoint_stream(
    source: AsyncIterator[Any], checkpointer: Any, thread_id: str
) -> AsyncGenerator[Any, None]:
    """Preserve an activity context across pull tasks used by v2/v3 streams.

    LangGraph v3 and Harness cancellation polling can advance a generator in a
    new task for every chunk. Reuse one Context for those sequential pulls so
    the async Memory scope and its saver children retain the same scope token.
    """
    context = copy_context()
    stream = _stream_with_activity(source, checkpointer, thread_id)
    try:
        while True:
            try:
                yield await asyncio.create_task(_advance(stream), context=context)
            except StopAsyncIteration:
                return
    finally:
        close = asyncio.create_task(_close_stream(stream), context=context)
        try:
            await asyncio.shield(close)
        except asyncio.CancelledError:
            await _settle(close)
            raise
