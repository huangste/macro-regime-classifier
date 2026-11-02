"""The app's redeploy safeguard: reload changed modules, never break imports.

Each test builds a throwaway package so the real ``regimelab`` modules are not
disturbed.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import reload_guard  # noqa: E402
from reload_guard import drop_stale_modules, reload_lock, stamp_modules  # noqa: E402

SLOW_MODULE = '''
from __future__ import annotations
import time
from dataclasses import dataclass
time.sleep({delay})

@dataclass(frozen=True)
class Fold:
    a: int

VALUE = {value!r}
'''


@pytest.fixture
def fake_package(tmp_path):
    """A package whose module takes a moment to import, like regimelab."""
    name = f"fakepkg_{uuid.uuid4().hex[:8]}"
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")

    def write(value: str, delay: float = 0.0) -> Path:
        path = pkg / "mod.py"
        path.write_text(SLOW_MODULE.format(delay=delay, value=value))
        return path

    sys.path.insert(0, str(tmp_path))
    yield name, write
    sys.path.remove(str(tmp_path))
    for n in [n for n in sys.modules if n == name or n.startswith(name + ".")]:
        del sys.modules[n]


def test_a_module_still_being_imported_is_never_dropped(fake_package):
    """The production failure: a second session ran the check while the first
    was importing, deleted the half-built module, and the import crashed."""
    name, write = fake_package
    write("v1", delay=0.5)
    errors = []

    def first_session():
        try:
            importlib.import_module(f"{name}.mod")
        except Exception as exc:  # pragma: no cover - the bug under test
            errors.append(exc)

    t = threading.Thread(target=first_session)
    t.start()
    time.sleep(0.2)
    assert drop_stale_modules(name) is False
    t.join()
    assert errors == []


def test_sessions_starting_together_all_import_cleanly(fake_package):
    name, write = fake_package
    write("v1", delay=0.3)
    errors = []

    def session():
        try:
            with reload_lock():
                drop_stale_modules(name)
                mod = importlib.import_module(f"{name}.mod")
                stamp_modules(name)
            assert mod.Fold(1).a == 1
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=session) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []


def test_a_changed_module_is_dropped_and_its_new_code_loaded(fake_package):
    """What the safeguard is for: a redeploy changes the library on disk."""
    name, write = fake_package
    path = write("v1")
    assert importlib.import_module(f"{name}.mod").VALUE == "v1"
    stamp_modules(name)

    write("v2")
    later = os.path.getmtime(path) + 5
    os.utime(path, (later, later))

    assert drop_stale_modules(name) is True
    assert f"{name}.mod" not in sys.modules
    assert importlib.import_module(f"{name}.mod").VALUE == "v2"


def test_an_unchanged_module_is_kept(fake_package):
    name, write = fake_package
    write("v1")
    mod = importlib.import_module(f"{name}.mod")
    stamp_modules(name)
    assert drop_stale_modules(name) is False
    assert sys.modules[f"{name}.mod"] is mod


def test_the_lock_survives_the_guard_being_reimported():
    """Streamlit re-imports app-folder modules after any change there; every
    session must still share one lock."""
    before = reload_lock()
    importlib.reload(reload_guard)
    assert reload_guard.reload_lock() is before
