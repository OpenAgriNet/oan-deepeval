import logging
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Mh_OANEvalClient:
    """
    HTTP client for the MH OAN evaluation service.

    Liveness check runs ONCE at construction.
    Each worker thread gets its own Session for true parallel connections.
    The static token is shared safely (read-only after init).

    Context-manager usage::

        with Mh_OANEvalClient(base_url=..., token=...) as client:
            response = client.chat("What crops grow in Karnataka?")
    """

    def __init__(
        self,
        base_url: str = "",
        token: str | None = None,
        liveness_retry_count: int = 5,
        liveness_retry_wait: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token                         # read-only after init — no lock needed
        self.liveness_retry_count = liveness_retry_count
        self.liveness_retry_wait = liveness_retry_wait

        # Each worker thread gets its own Session via this
        self._thread_local = threading.local()

        # Liveness runs ONCE here, not per-thread
        self._wait_for_liveness()

    # ------------------------------------------------------------------
    # Thread-local Session
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Create a session with connection pooling and idempotent retries."""
        session = requests.Session()
        retry = Retry(
            total=3,
            status_forcelist={502, 503, 504},
            allowed_methods={"GET"},
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        if self.token:
            session.headers["Authorization"] = f"Bearer {self.token}"
        return session

    @property
    def _session(self) -> requests.Session:
        """One Session per worker thread — parallel connection pools, no contention."""
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = self._build_session()
            logger.debug(
                "New session created for thread %s",
                threading.current_thread().name,
            )
        return self._thread_local.session

    # ------------------------------------------------------------------
    # Liveness — runs once at __init__, never again
    # ------------------------------------------------------------------

    def _wait_for_liveness(self) -> None:
        url = f"{self.base_url}/api/health/live"
        # Use a plain one-off session so thread-local isn't touched yet
        probe = requests.Session()
        try:
            for attempt in range(1, self.liveness_retry_count + 1):
                try:
                    r = probe.get(url, headers={"Accept": "application/json"}, timeout=5)
                    if r.status_code in (200, 403):
                        logger.info(
                            "[Mh_OANEvalClient] Service is live — HTTP %s (attempt %d)",
                            r.status_code, attempt,
                        )
                        return
                    logger.warning(
                        "[Mh_OANEvalClient] Liveness: HTTP %s (attempt %d/%d)",
                        r.status_code, attempt, self.liveness_retry_count,
                    )
                except requests.RequestException as exc:
                    logger.warning(
                        "[Mh_OANEvalClient] Liveness error: %s (attempt %d/%d)",
                        exc, attempt, self.liveness_retry_count,
                    )
                if attempt < self.liveness_retry_count:
                    time.sleep(self.liveness_retry_wait)
        finally:
            probe.close()   # discard the probe session immediately

        raise RuntimeError(
            f"[Mh_OANEvalClient] Service did not become live after "
            f"{self.liveness_retry_count} attempts."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        session_id: str = "eval-session",
        user_id: str = "eval-user",
        source_lang: str = "en",
        target_lang: str = "en",
    ) -> str | None:
        """
        Send a chat query and return the full streamed response as a string.
        Runs fully in parallel — no locks held during the HTTP call.
        """
        url = f"{self.base_url}/api/chat/"
        params = {
            "query": query,
            "session_id": session_id,
            "user_id": user_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        try:
            # _session is thread-local — no lock needed
            response = self._session.get(url, params=params, stream=True, timeout=60)
        except requests.RequestException as exc:
            logger.error("[Mh_OANEvalClient] Chat request raised: %s", exc)
            return None

        if response.status_code != 200:
            logger.error("[Mh_OANEvalClient] Chat failed: HTTP %s", response.status_code)
            return None

        raw = bytearray()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                raw.extend(chunk)

        return raw.decode("utf-8", errors="replace").strip() or None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close this thread's session if one was opened."""
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            session.close()
            del self._thread_local.session
        logger.debug("[Mh_OANEvalClient] Session closed for thread %s", threading.current_thread().name)

    shutdown = close  # backward compatibility

    def __enter__(self) -> "Mh_OANEvalClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()