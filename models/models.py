from __future__ import annotations

import uuid
from dataclasses import dataclass, field


LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "as": "Assamese (অসমীয়া)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "mr": "Marathi (मराठी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
}


@dataclass
class OANTestCase:
    name: str
    category: str
    language: str
    input: str
    expected: str | None = None
    is_decline: bool = False
    context: list[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"eval-{uuid.uuid4().hex}")
    section: str = ""
