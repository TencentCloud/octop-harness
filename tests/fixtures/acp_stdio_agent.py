#!/usr/bin/env python3
"""Stdio ACP agent for acp-tck / local IDE smoke tests (offline mock harness).

Run::

    uv run python tests/fixtures/acp_stdio_agent.py

Against acp-tck (from an acp-tck checkout)::

    uv run acp-tck --agent-cwd /tmp/acp-tck-wd --timeout 30 \\
      -- uv run --project /path/to/octop-harness \\
         python /path/to/octop-harness/tests/fixtures/acp_stdio_agent.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any


class _MockHarness:
    """Deterministic offline stream for protocol tests (no LLM)."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s

    async def stream(self, request: dict[str, Any]) -> Any:
        text = ""
        for msg in request.get("messages") or []:
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = str(msg.get("content") or "")
                break
        # Yield between short sleeps so session/cancel can interrupt (ACP-CANCEL-*).
        if self._delay_s > 0:
            steps = max(1, int(self._delay_s / 0.1))
            for _ in range(steps):
                await asyncio.sleep(0.1)
                yield {"type": "token", "content": "."}
        yield {"type": "token", "content": text or "ok"}


async def _main() -> None:
    from octop_harness.acp.server import ACPExpertOption, run_harness_acp_server

    delay = float(os.environ.get("OCTOP_ACP_TCK_PROMPT_DELAY", "0") or "0")
    multi = os.environ.get("OCTOP_ACP_TCK_MULTI_EXPERT", "").strip() in {"1", "true", "yes"}
    main = _MockHarness(delay_s=delay)
    if multi:
        ops = _MockHarness(delay_s=delay)
        await run_harness_acp_server(
            main,
            agent_id="main",
            experts=[
                ACPExpertOption(value="main", name="Main", description="default"),
                ACPExpertOption(value="ops", name="Ops"),
            ],
            resolve_harness=lambda eid: ops if eid == "ops" else main,
        )
    else:
        await run_harness_acp_server(main, agent_id="main")


if __name__ == "__main__":
    asyncio.run(_main())
