"""Ask the user, or let the external agent run without an approval card.

``yolo`` selects an ACP session mode whose id or name looks like YOLO / bypass
when the agent advertises one, and answers any permission request that still
arrives with an allow option. Destructive commands the adapter hard-blocks
stay cancelled.
"""

from __future__ import annotations

from typing import Any

PermissionMode = str

# ``dontAsk`` is the opposite: it denies actions that were not pre-approved.
_YOLO_MARKERS = (
    "yolo",
    "bypass",
    "full-access",
    "fullaccess",
    "auto-accept",
    "autoaccept",
    "unrestricted",
    "danger",
)
_ASK_MARKERS = ("default", "ask", "manual", "prompt", "normal", "code", "craft")


def parse_permission_mode(value: object) -> PermissionMode:
    return "yolo" if value == "yolo" else "ask"


def _blob(mode: Any) -> str:
    mode_id = str(getattr(mode, "id", "") or "")
    name = str(getattr(mode, "name", "") or "")
    if isinstance(mode, dict):
        mode_id = str(mode.get("id") or mode_id)
        name = str(mode.get("name") or name)
    return f"{mode_id} {name}".strip().lower()


def _mode_id(mode: Any) -> str:
    if isinstance(mode, dict):
        return str(mode.get("id") or "").strip()
    return str(getattr(mode, "id", "") or "").strip()


def _is_yolo(text: str) -> bool:
    folded = text.lower()
    return any(marker in folded for marker in _YOLO_MARKERS)


def _is_deny(text: str) -> bool:
    """``dontAsk`` denies instead of prompting. ``ask`` is a substring of it."""
    folded = text.lower().replace("'", "").replace("\u2019", "")
    return "dontask" in folded or "dont ask" in folded


def is_unattended_mode(value: object) -> bool:
    """True when a session mode id means the agent should run without asking."""
    return _is_yolo(str(value or ""))


def select_session_mode(
    available: list[Any],
    permission_mode: str,
    session_mode: str = "",
    default_mode_id: str | None = None,
) -> tuple[str, str | None]:
    """Return ``(ask|yolo, mode id to set)``.

    An explicit id is used when the agent lists it, or when the agent listed
    no modes at all. The id skips approval when it is unattended, and also
    when the caller already asked for yolo. Otherwise fall back to the
    yolo/ask match.
    """
    explicit = session_mode.strip()
    ids = [mode_id for mode in available if (mode_id := _mode_id(mode))]
    if explicit and (not ids or explicit in ids):
        unattended = _is_yolo(explicit) or parse_permission_mode(permission_mode) == "yolo"
        return ("yolo" if unattended else "ask"), explicit
    selected = parse_permission_mode(permission_mode)
    if selected == "yolo":
        return "yolo", yolo_mode_id(available)
    return "ask", ask_mode_id(available, default_mode_id)


def yolo_mode_id(modes: list[Any]) -> str | None:
    for mode in modes:
        mode_id = _mode_id(mode)
        if mode_id and _is_yolo(_blob(mode)):
            return mode_id
    return None


def ask_mode_id(modes: list[Any], default_mode_id: str | None) -> str | None:
    """Mode to restore when the user wants approvals again."""
    if default_mode_id and not _is_yolo(default_mode_id) and not _is_deny(default_mode_id):
        return default_mode_id
    for mode in modes:
        blob = _blob(mode)
        mode_id = _mode_id(mode)
        if mode_id and not _is_yolo(blob) and not _is_deny(blob) and any(marker in blob for marker in _ASK_MARKERS):
            return mode_id
    return None


def mode_state(response: Any) -> tuple[str | None, list[Any], str | None]:
    """``(current, available, default)`` from a new/load session response."""
    modes = getattr(response, "modes", None)
    if modes is None and isinstance(response, dict):
        modes = response.get("modes")
    if modes is None:
        return None, [], None
    if isinstance(modes, dict):
        current = str(modes.get("currentModeId") or modes.get("current_mode_id") or "").strip()
        available = list(modes.get("availableModes") or modes.get("available_modes") or [])
    else:
        current = str(getattr(modes, "current_mode_id", "") or "").strip()
        available = list(getattr(modes, "available_modes", None) or [])
    current_id = current or None
    default_id = ask_mode_id(available, current_id)
    return current_id, available, default_id


def yolo_option_id(options: list[Any]) -> str:
    """Allow option for an unattended turn. Prefer a permanent allow."""
    always = ""
    plain = ""
    once = ""
    for option in options:
        if not isinstance(option, dict):
            continue
        option_id = str(option.get("optionId") or option.get("option_id") or "").strip()
        label = str(option.get("kind") or option.get("name") or option.get("title") or "").lower()
        if not option_id or any(token in label for token in ("reject", "cancel", "deny")):
            continue
        if "always" in label:
            always = always or option_id
        elif "once" in label:
            once = once or option_id
        else:
            plain = plain or option_id
    return always or plain or once
