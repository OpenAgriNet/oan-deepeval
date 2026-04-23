from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from settings.config import BASE_URL, MAX_WORKERS
from models.models import OANTestCase
from client.oan_eval_client import OANEvalClient
from client.mh_oan_eval_client import Mh_OANEvalClient

# Thread-local storage — each worker thread gets its own client instances
_thread_local = threading.local()


def _get_thread_client(base_url: str | None, api_key: str | None) -> OANEvalClient:
    """Return a per-thread OANEvalClient, creating one if needed."""
    desired_url = (base_url or BASE_URL).rstrip("/")

    client: OANEvalClient | None = getattr(_thread_local, "oan_client", None)

    if client is None or client.base_url != desired_url or (api_key and client.api_key != api_key):
        _thread_local.oan_client = OANEvalClient(base_url=desired_url, api_key=api_key)

    return _thread_local.oan_client


def _get_thread_mh_client(base_url: str, token: str) -> Mh_OANEvalClient:
    """Return a per-thread Mh_OANEvalClient, creating one if needed."""
    desired_url = base_url.rstrip("/")

    client: Mh_OANEvalClient | None = getattr(_thread_local, "mh_client", None)

    if client is None or client.base_url != desired_url or client.token != token:
        if client is not None:
            client.close()
        _thread_local.mh_client = Mh_OANEvalClient(base_url=desired_url, token=token)

    return _thread_local.mh_client


def _call_api(tc: OANTestCase, base_url: str | None, api_key: str | None) -> tuple[str, str | None]:
    output = _get_thread_client(base_url, api_key).chat(
        query=tc.input,
        session_id=tc.session_id,
        user_id="eval-user",
        source_lang=tc.language,
        target_lang=tc.language,
    )
    return tc.name, output


def _call_mh_api(tc: OANTestCase, base_url: str, token: str) -> tuple[str, str | None]:
    output = _get_thread_mh_client(base_url, token).chat(
        query=tc.input,
        session_id=tc.session_id,
        user_id="eval-user",
        source_lang=tc.language,
        target_lang=tc.language,
    )
    return tc.name, output


def fetch_all_outputs(
    cases: list[OANTestCase],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    max_workers: int | None = None,
) -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    worker_count = max_workers or MAX_WORKERS

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_call_api, tc, base_url, api_key): tc for tc in cases}
        for future in as_completed(futures):
            tc = futures[future]
            try:
                name, output = future.result()
                results[name] = output
                print(f"[API] OK {name!r} -> {len(output or '')} chars")
            except Exception as exc:
                results[tc.name] = None
                print(f"[API] FAIL {tc.name!r} -> {exc}")

    return results


def fetch_all_mh_outputs(
    cases: list[OANTestCase],
    *,
    base_url: str,
    token: str,
    max_workers: int | None = None,
) -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    worker_count = max_workers or MAX_WORKERS

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_call_mh_api, tc, base_url, token): tc for tc in cases}
        for future in as_completed(futures):
            tc = futures[future]
            try:
                name, output = future.result()
                results[name] = output
                print(f"[MH-API] OK {name!r} -> {len(output or '')} chars")
            except Exception as exc:
                results[tc.name] = None
                print(f"[MH-API] FAIL {tc.name!r} -> {exc}")

    return results