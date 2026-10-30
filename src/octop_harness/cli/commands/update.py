"""The `octop-harness update` command — self-upgrade the CLI."""

from __future__ import annotations

import subprocess
import sys

import click
from rich.console import Console

console = Console()

_PIP_INSTALL_TIMEOUT = 600


@click.command()
@click.option("--pre", is_flag=True, default=False, help="Include pre-release versions.")
def update(pre: bool) -> None:
    """Upgrade octop-harness-cli to the latest version from PyPI."""
    console.print("[dim]Checking for updates...[/]")

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "octop-harness-cli"]
    if pre:
        cmd.append("--pre")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=_PIP_INSTALL_TIMEOUT,
        )
        if result.returncode == 0:
            # Extract version info from pip output
            if "already satisfied" in result.stdout.lower() or "already up-to-date" in result.stdout.lower():
                console.print("[green]✓[/] Already up to date.")
            else:
                console.print("[green]✓[/] Updated successfully.")
                console.print(f"[dim]{result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''}[/]")
        else:
            console.print(f"[red]Update failed:[/] {result.stderr.strip()}")
            raise SystemExit(1)
    except subprocess.TimeoutExpired:
        console.print(f"[red]Error:[/] pip install timed out after {_PIP_INSTALL_TIMEOUT}s")
        raise SystemExit(1) from None
    except FileNotFoundError:
        console.print("[red]Error:[/] pip not found. Please upgrade manually: pip install --upgrade octop-harness-cli")
        raise SystemExit(1) from None
