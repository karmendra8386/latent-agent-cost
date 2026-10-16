"""Network-environment hygiene for this VM.

All egress here goes through an HTTP proxy (``http_proxy`` /
``https_proxy`` env vars), and ``no_proxy`` contains bracketed
IPv6 entries such as ``[::1]``. ``httpx`` (used by
``huggingface_hub``) parses every ``no_proxy`` entry as a URL
pattern and crashes on those entries
(``InvalidURL: Invalid port: ':1]'``), which makes
``datasets.load_dataset`` and ``transformers`` downloads fail
before any network traffic happens.

``sanitize_proxy_env()`` drops the colon-bearing ``no_proxy``
entries (IPv6 literals) for this process only. Every fetch this
project makes is to a public host through the proxy, so no
behaviour changes -- local addresses are never fetched.
"""

from __future__ import annotations

import os


def sanitize_proxy_env() -> None:
    for key in ("no_proxy", "NO_PROXY"):
        raw = os.environ.get(key)
        if not raw:
            continue
        kept = [e.strip() for e in raw.split(",")
                if e.strip() and ":" not in e and "[" not in e]
        os.environ[key] = ",".join(kept)
