from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from lifetwin.data.beep import normalize_fastcharge_protocol


_VERSION_PATTERN = re.compile(rb'@version"\s*:\s*("(?:\\.|[^"\\])*")')


@dataclass(frozen=True)
class BeepIdentityRecord:
    barcode: str
    channel_id: int
    source_domain: str
    batch_id: str
    protocol_raw: str
    protocol_id: str
    beep_version: str
    source_filename: str
    source_size_bytes: int


def _read_identity_header(
    path: Path,
    *,
    max_header_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """Read only the top-level bytes before the BEEP summary key."""
    marker = b'"summary"'
    buffer = bytearray()
    with path.open("rb") as stream:
        while len(buffer) < max_header_bytes:
            value = stream.read(1)
            if not value:
                raise ValueError(f"Missing BEEP summary boundary: {path}")
            buffer.extend(value)
            if buffer.endswith(marker):
                prefix = bytes(buffer[: -len(marker)]).rstrip()
                if not prefix.endswith(b","):
                    raise ValueError(f"Invalid BEEP identity boundary: {path}")
                return json.loads(prefix[:-1] + b"}")
    raise ValueError(f"BEEP identity header exceeds safe limit: {path}")


def _read_tail_version(path: Path, *, max_tail_bytes: int = 64 * 1024) -> str:
    """Read the trailing BEEP version metadata without parsing array values."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - max_tail_bytes))
        tail = stream.read()
    matches = list(_VERSION_PATTERN.finditer(tail))
    if not matches:
        return "unknown"
    return str(json.loads(matches[-1].group(1)))


def read_beep_identity(path: str | Path) -> BeepIdentityRecord:
    """Return only whitelisted identity fields, never summary or curve values."""
    source_path = Path(path)
    value = _read_identity_header(source_path)
    if value.get("@module") != "beep.structure":
        raise ValueError(f"Unexpected BEEP module in {source_path}")
    if value.get("@class") != "ProcessedCyclerRun":
        raise ValueError(f"Unexpected BEEP class in {source_path}")
    barcode = str(value.get("barcode") or "").strip().upper()
    protocol_raw = str(value.get("protocol") or "").strip()
    if not barcode or not protocol_raw:
        raise ValueError(f"BEEP barcode and protocol are required: {source_path}")
    source_domain, batch_id, protocol_id = normalize_fastcharge_protocol(protocol_raw)
    try:
        channel_id = int(value["channel_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid BEEP channel_id: {source_path}") from exc
    return BeepIdentityRecord(
        barcode=barcode,
        channel_id=channel_id,
        source_domain=source_domain,
        batch_id=batch_id,
        protocol_raw=protocol_raw,
        protocol_id=protocol_id,
        beep_version=_read_tail_version(source_path),
        source_filename=source_path.name,
        source_size_bytes=source_path.stat().st_size,
    )
