"""tests/test_acp_service.py - unit tests for ACP service scheduling."""

from __future__ import annotations

import asyncio
from typing import Any

from octop_harness.acp import service as acp_service


class _StubService:
    async def close_all_sessions(self) -> None:
        await self._release.wait()


def test_schedule_close_keeps_task_reference() -> None:
    """A scheduled close must stay referenced until it completes."""

    async def scenario() -> None:
        stub = _StubService()
        stub._release = asyncio.Event()
        acp_service._schedule_close(stub)  # type: ignore[arg-type]
        pending = [t for t in acp_service._CLOSE_TASKS if not t.done()]
        assert pending, "scheduled close task was not retained"
        task = pending[0]
        stub._release.set()
        await task
        await asyncio.sleep(0)
        assert task not in acp_service._CLOSE_TASKS

    asyncio.run(scenario())
