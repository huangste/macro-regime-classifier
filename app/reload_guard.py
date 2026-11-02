"""Reload the research library safely when a redeploy changes it.

Streamlit re-imports changed modules only from the app's own folder.
``regimelab`` sits beside that folder, so after a redeploy the running process
would keep the old library and the new script could fail to import from it.
The app therefore drops stale ``regimelab`` modules itself before importing.

Doing that safely needs three rules.  The first version had only the idea, and
failed in production when two sessions started together: one session deleted
a module the other was still importing.

* A module is stale only if it was stamped after import and its source file
  has changed since.  An unstamped module is one still being imported (or
  imported lazily), and is never touched.
* A module whose import is in progress is never deleted.
* The check, the imports and the stamping run under one lock shared by every
  session in the process, so concurrent sessions cannot interleave them.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading

_LOCK_ATTR = "_project_reload_lock"
_STAMP_ATTR = "_source_mtime"


def reload_lock() -> threading.RLock:
    """The process-wide lock.

    Kept on ``sys`` because Streamlit re-imports this module whenever a file in
    the app folder changes, which would otherwise hand each session a fresh,
    useless lock.  ``dict.setdefault`` is atomic, so sessions starting at the
    same moment still receive the same lock.
    """
    return sys.__dict__.setdefault(_LOCK_ATTR, threading.RLock())


def _modules(prefix: str) -> list[str]:
    return [n for n in list(sys.modules) if n == prefix or n.startswith(prefix + ".")]


def _importing(mod) -> bool:
    # importlib sets this on a module's spec while its code is executing.
    return bool(getattr(getattr(mod, "__spec__", None), "_initializing", False))


def drop_stale_modules(prefix: str) -> bool:
    """Remove every ``prefix`` module if any of them changed on disk.

    Returns True if modules were dropped, so the caller can clear caches that
    hold objects built from the old code.
    """
    names = _modules(prefix)
    stale = False
    for n in names:
        mod = sys.modules.get(n)
        path = getattr(mod, "__file__", None)
        stamp = getattr(mod, _STAMP_ATTR, None)
        if mod is None or path is None or stamp is None or _importing(mod):
            continue
        if not os.path.exists(path) or os.path.getmtime(path) != stamp:
            stale = True
            break
    if not stale:
        return False
    for n in names:
        mod = sys.modules.get(n)
        if mod is not None and not _importing(mod):
            sys.modules.pop(n, None)
    importlib.invalidate_caches()
    return True


def stamp_modules(prefix: str) -> None:
    """Record the source time of every fully imported ``prefix`` module."""
    for n in _modules(prefix):
        mod = sys.modules.get(n)
        path = getattr(mod, "__file__", None)
        if (mod is not None and path and not _importing(mod)
                and getattr(mod, _STAMP_ATTR, None) is None):
            setattr(mod, _STAMP_ATTR, os.path.getmtime(path))
