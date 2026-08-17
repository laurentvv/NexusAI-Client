"""Utility functions for NexusAI-Client, including image and multimedia processing."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import BinaryIO


def detect_mime_type_from_bytes(data: bytes) -> str:
    """Inspect magic numbers to deduce image/document MIME type."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"%PDF"):
        return "application/pdf"
    return "image/jpeg"


def load_image_as_base64_and_mime(
    image: str | Path | bytes | BinaryIO,
) -> tuple[str, str]:
    """Load image from file path, Path object, raw bytes, or base64 string.

    Returns:
        tuple[str, str]: (raw_base64_str, mime_type)
    """
    # 1. Raw bytes
    if isinstance(image, bytes):
        mime = detect_mime_type_from_bytes(image)
        b64_str = base64.b64encode(image).decode("utf-8")
        return b64_str, mime

    # 2. File-like object
    if hasattr(image, "read"):
        content = image.read()  # type: ignore[union-attr]
        if isinstance(content, str):
            content = content.encode("utf-8")
        mime = detect_mime_type_from_bytes(content)
        return base64.b64encode(content).decode("utf-8"), mime

    # 3. Pathlib Path or string path
    if isinstance(image, Path) or (
        isinstance(image, str)
        and not image.startswith(("data:", "http://", "https://"))
    ):
        p = Path(image)
        if p.exists() and p.is_file():
            raw_bytes = p.read_bytes()
            mime, _ = mimetypes.guess_type(str(p))
            if not mime:
                mime = detect_mime_type_from_bytes(raw_bytes)
            return base64.b64encode(raw_bytes).decode("utf-8"), mime

    # 4. Data URI string (e.g. data:image/png;base64,iVBORw0KGgo...)
    if isinstance(image, str) and image.startswith("data:"):
        header, _, b64_part = image.partition(",")
        mime = header.removeprefix("data:").removesuffix(";base64")
        return b64_part.strip(), mime

    # 5. Raw base64 string fallback
    if isinstance(image, str):
        try:
            decoded = base64.b64decode(image, validate=True)
            mime = detect_mime_type_from_bytes(decoded)
            return image.strip(), mime
        except Exception:
            pass

    raise ValueError(
        f"Unable to resolve image source: {type(image)}. "
        "Expected valid file path, Path, bytes, data-URI, or URL."
    )


def load_image_as_data_uri(image: str | Path | bytes | BinaryIO) -> str:
    """Format image as standard base64 data URI (data:<mime>;base64,<data>)."""
    if isinstance(image, str) and (
        image.startswith("http://")
        or image.startswith("https://")
        or image.startswith("data:")
    ):
        return image
    b64_str, mime = load_image_as_base64_and_mime(image)
    return f"data:{mime};base64,{b64_str}"
