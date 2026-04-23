import json
import time
import threading
from dotenv import load_dotenv
import requests
load_dotenv()


class OANEvalClient:
    def __init__(
        self,
        base_url: str = "",
        api_key: str | None = None,
        liveness_retry_count: int = 5,
        liveness_retry_wait: float = 3.0,
        token_refresh_buffer: float = 60.0,
        token_params: dict = {
            "mobile": "9876543212",
            "name": "OAN Eval Client",
            "role": "Evaluvater",
            "metadata": "v1.0"
        },
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.liveness_retry_count = liveness_retry_count
        self.liveness_retry_wait = liveness_retry_wait
        self.token_refresh_buffer = token_refresh_buffer
        self.token_params = token_params

        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._lock = threading.Lock()
        self._refresh_timer: threading.Timer | None = None

        # Each worker thread gets its own Session — real parallel connections
        self._thread_local = threading.local()

        # These run ONCE at construction time, not per-thread
        self._wait_for_liveness()
        if self.api_key:
            self._token = self.api_key
            self._token_expiry = time.time() + (365 * 24 * 60 * 60)
        else:
            self._refresh_token()

    # ------------------------------------------------------------------
    # Thread-local Session
    # ------------------------------------------------------------------

    @property
    def _session(self) -> requests.Session:
        """One requests.Session per thread — gives parallel connection pools."""
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = requests.Session()
            print(f"[OANEvalClient] New session for thread {threading.current_thread().name}")
        return self._thread_local.session

    # ------------------------------------------------------------------
    # Liveness — called once at __init__
    # ------------------------------------------------------------------

    def _wait_for_liveness(self) -> None:
        url = f"{self.base_url}/api/health/live"
        for attempt in range(1, self.liveness_retry_count + 1):
            try:
                resp = requests.get(url, headers={"accept": "application/json"}, timeout=5)
                if resp.status_code in (200, 403):
                    print(f"[OANEvalClient] Service is live (attempt {attempt})")
                    return
                print(f"[OANEvalClient] Liveness: status {resp.status_code} (attempt {attempt}/{self.liveness_retry_count})")
            except requests.RequestException as e:
                print(f"[OANEvalClient] Liveness error: {e} (attempt {attempt}/{self.liveness_retry_count})")

            if attempt < self.liveness_retry_count:
                time.sleep(self.liveness_retry_wait)

        raise RuntimeError(
            f"[OANEvalClient] Service did not become live after {self.liveness_retry_count} attempts."
        )

    # ------------------------------------------------------------------
    # Token management — one shared token, background refresh
    # ------------------------------------------------------------------

    def _fetch_token(self) -> tuple[str, float]:
        url = f"{self.base_url}/api/token"
        resp = requests.post(
            url,
            params=self.token_params,
            headers={"accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data["token"]
        expires_in: int = data.get("expires_in", 900)
        return token, time.time() + expires_in

    def _refresh_token(self) -> None:
        with self._lock:
            token, expiry = self._fetch_token()
            self._token = token
            self._token_expiry = expiry
            print(f"[OANEvalClient] Token refreshed — expires in {int(expiry - time.time())}s")

        if self._refresh_timer is not None:
            self._refresh_timer.cancel()

        refresh_in = max((self._token_expiry - time.time()) - self.token_refresh_buffer, 5)
        self._refresh_timer = threading.Timer(refresh_in, self._refresh_token)
        self._refresh_timer.daemon = True
        self._refresh_timer.start()
        print(f"[OANEvalClient] Next token refresh in {int(refresh_in)}s")

    @property
    def token(self) -> str:
        with self._lock:
            if self._token is None:
                raise RuntimeError("[OANEvalClient] Token not initialized")
            if time.time() >= self._token_expiry - self.token_refresh_buffer:
                print("[OANEvalClient] Token near expiry — refreshing synchronously")
                self._refresh_token()
            return self._token

    # ------------------------------------------------------------------
    # Chat — uses shared token, per-thread session
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        session_id: str = "eval-session",
        user_id: str = "eval-user",
        source_lang: str = "en",
        target_lang: str = "en",
    ) -> str | None:
        resp = self._session.get(        # <-- thread-local session
            f"{self.base_url}/api/chat/",
            params={
                "query": query,
                "session_id": session_id,
                "user_id": user_id,
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
            headers={"Authorization": f"Bearer {self.token}"},  # <-- shared token
            stream=True,
            timeout=60,
        )

        if resp.status_code != 200:
            print(f"[OANEvalClient] Chat failed: {resp.status_code}")
            return None

        raw = bytearray()
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                raw.extend(chunk)

        return raw.decode("utf-8", errors="replace").strip() or None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.cancel()
        print("[OANEvalClient] Shutdown complete.")