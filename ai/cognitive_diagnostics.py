"""Single-file local debug trace for one Cognitive Core test session."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel


_LOGGER = logging.getLogger(__name__)
DIAGNOSTICS_FLAG = "COGNITIVE_CORE_DIAGNOSTICS_ENABLED"
DIAGNOSTICS_PATH = "COGNITIVE_CORE_DIAGNOSTICS_PATH"
_SESSION_ID = f"process-{uuid4().hex}"
_WRITE_LOCK = threading.Lock()


def diagnostics_enabled() -> bool:
    """Enable the local development trace by default; explicit false disables it."""
    raw = os.getenv(DIAGNOSTICS_FLAG)
    if raw is None or not raw.strip():
        return True
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    _LOGGER.warning("Unrecognized %s=%r; diagnostics remain enabled.", DIAGNOSTICS_FLAG, raw)
    return True


def diagnostics_path() -> Path:
    configured = os.getenv(DIAGNOSTICS_PATH)
    if configured and configured.strip():
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "diagnostics" / "latest_test_dialog.json"


def conversation_identifier(user_id: Any) -> str:
    if user_id is None:
        return f"stateless-{_SESSION_ID}"
    digest = hashlib.sha256(f"telegram:{user_id}".encode("utf-8")).hexdigest()
    return f"telegram-{digest[:16]}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    return repr(value)


def _new_document(
    conversation_id: str,
    *,
    trigger: str,
    start_exchange: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = utc_timestamp()
    return {
        "diagnostic_schema": "cognitive-core-test-session.v2",
        "session": {
            "session_id": str(uuid4()),
            "process_session_id": _SESSION_ID,
            "conversation_id": conversation_id,
            "started_at": started_at,
            "last_updated_at": started_at,
            "start_trigger": trigger,
            "start_exchange": json_safe(start_exchange),
        },
        "turns": [],
        "events": [],
    }


def _read_document(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def start_diagnostic_session(
    user_id: Any,
    *,
    trigger: str = "telegram_start",
    start_exchange: dict[str, Any] | None = None,
) -> Path | None:
    """Replace the prior test trace with one clean, explicitly started session."""
    if not diagnostics_enabled():
        return None
    try:
        destination = diagnostics_path()
        document = _new_document(
            conversation_identifier(user_id),
            trigger=trigger,
            start_exchange=start_exchange,
        )
        with _WRITE_LOCK:
            _atomic_write(destination, document)
        return destination
    except Exception:
        _LOGGER.warning("Could not start Cognitive diagnostic session.", exc_info=True)
        return None


def _current_document(destination: Path, conversation_id: str) -> dict[str, Any]:
    document = _read_document(destination)
    session = document.get("session", {}) if document else {}
    if (
        not document
        or session.get("conversation_id") != conversation_id
        or session.get("process_session_id") != _SESSION_ID
    ):
        return _new_document(conversation_id, trigger="implicit_first_turn")
    return document


def write_turn_diagnostic(record: dict[str, Any]) -> Path | None:
    """Atomically upsert one real turn; diagnostic failure never breaks consultation."""
    if not diagnostics_enabled():
        return None
    try:
        destination = diagnostics_path()
        safe_record = json_safe(record)
        conversation_id = safe_record.get("conversation_id")
        turn_id = safe_record.get("turn_id")
        if not isinstance(conversation_id, str) or not isinstance(turn_id, str):
            raise ValueError("Diagnostic record requires conversation_id and turn_id")
        safe_record["recorded_at"] = utc_timestamp()
        with _WRITE_LOCK:
            document = _current_document(destination, conversation_id)
            turns = document.setdefault("turns", [])
            for index, existing in enumerate(turns):
                if existing.get("turn_id") == turn_id:
                    turns[index] = safe_record
                    break
            else:
                turns.append(safe_record)
            document["session"]["last_updated_at"] = utc_timestamp()
            _atomic_write(destination, document)
        return destination
    except Exception:
        _LOGGER.warning("Cognitive diagnostic write failed; continuing without diagnostics.", exc_info=True)
        return None


def record_telegram_delivery(
    user_id: Any,
    turn_id: str | None,
    *,
    reply: str,
    delivered: bool,
    error: str | None = None,
    telegram_metadata: dict[str, Any] | None = None,
) -> Path | None:
    """Attach the observable Telegram delivery result to an already captured turn."""
    if not diagnostics_enabled():
        return None
    try:
        destination = diagnostics_path()
        conversation_id = conversation_identifier(user_id)
        with _WRITE_LOCK:
            document = _current_document(destination, conversation_id)
            matched = False
            turns = document.setdefault("turns", [])
            for turn in reversed(turns):
                if turn_id is None or turn.get("turn_id") == turn_id:
                    turn_id = turn.get("turn_id")
                    turn["telegram_delivery"] = {
                        "attempted_at": utc_timestamp(),
                        "reply": reply,
                        "delivered": delivered,
                        "error": error,
                        "metadata": json_safe(telegram_metadata),
                    }
                    matched = True
                    break
            if not matched:
                document.setdefault("events", []).append(
                    {
                        "event": "telegram_delivery_without_turn_record",
                        "at": utc_timestamp(),
                        "turn_id": turn_id,
                        "reply": reply,
                        "delivered": delivered,
                        "error": error,
                        "metadata": json_safe(telegram_metadata),
                    }
                )
            document["session"]["last_updated_at"] = utc_timestamp()
            _atomic_write(destination, document)
        return destination
    except Exception:
        _LOGGER.warning("Telegram delivery diagnostic update failed.", exc_info=True)
        return None


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="store_true", help="Print the diagnostic file path.")
    parser.add_argument("--show", action="store_true", help="Print the current diagnostic JSON.")
    parser.add_argument("--purge", action="store_true", help="Delete the current test trace.")
    args = parser.parse_args()
    path = diagnostics_path()
    if args.purge:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        print(f"Removed local diagnostic trace: {path}")
        return 0
    if args.show:
        print(path.read_text(encoding="utf-8") if path.exists() else "No diagnostic trace.")
        return 0
    if args.path:
        print(path)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
