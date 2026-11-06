"""
test_embedder_threadsafety.py — get_vector_store() under concurrent first calls
================================================================================
Live regression: an agent issuing several searches in parallel hit a cold cache from
many threads at once; each built its own ChromaDB client and the process crashed
("RustBindingsAPI object has no attribute 'bindings'"). Offline: Chroma is faked.
"""

import contextlib
import threading
import time

import pytest

from src.rag import embedder

THREADS = 8


class SlowChroma:
    """Stand-in for Chroma whose construction takes long enough for threads to overlap."""

    constructed = 0
    active = 0
    max_active = 0
    _guard = threading.Lock()

    def __init__(self, **kwargs) -> None:
        with SlowChroma._guard:
            SlowChroma.constructed += 1
            SlowChroma.active += 1
            SlowChroma.max_active = max(SlowChroma.max_active, SlowChroma.active)
        time.sleep(0.05)
        with SlowChroma._guard:
            SlowChroma.active -= 1


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch: pytest.MonkeyPatch):
    SlowChroma.constructed = SlowChroma.active = SlowChroma.max_active = 0
    monkeypatch.setattr(embedder, "Chroma", SlowChroma)
    monkeypatch.setattr(embedder, "OpenAIEmbeddings", lambda **kw: object())
    embedder.get_vector_store.cache_clear()
    yield
    embedder.get_vector_store.cache_clear()


def hammer() -> list:
    """Call get_vector_store from THREADS threads released at the same instant."""
    barrier = threading.Barrier(THREADS)
    results: list = []

    def worker() -> None:
        barrier.wait()
        results.append(embedder.get_vector_store())

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_first_calls_build_the_store_exactly_once() -> None:
    results = hammer()

    assert SlowChroma.constructed == 1
    assert SlowChroma.max_active == 1  # never two constructions overlapping
    assert len(results) == THREADS and all(r is results[0] for r in results)


def test_this_test_would_catch_the_race_without_the_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sensitivity check: with the lock removed, the same hammering DOES build the store
    more than once, so the test above is not passing by luck."""
    monkeypatch.setattr(embedder, "_build_lock", contextlib.nullcontext())

    hammer()

    assert SlowChroma.constructed > 1


def test_later_calls_reuse_the_cached_store() -> None:
    first = embedder.get_vector_store()
    assert embedder.get_vector_store() is first
    assert SlowChroma.constructed == 1
