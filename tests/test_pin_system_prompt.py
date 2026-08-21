# SPDX-License-Identifier: Apache-2.0
"""--pin-system-prompt wiring for the batched lane.

The flag's original target was the removed SimpleEngine's trie cache
(``engine._prefix_cache``) — on BatchedEngine it silently no-oped. The
rewire routes it through ``MemoryAwarePrefixCache.pin_prefix``: the
engine registers a pending pin for the rendered system-segment prefix,
the scheduler's boundary snapshot stores that exact key, and the cache
marks the entry ``protected`` so LRU / prefix-subset / hybrid-bound
eviction never reclaim it.

These tests exercise the cache mechanics with fakes — no model load.
"""

from __future__ import annotations

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig


class _FakeCacheLayer:
    """Minimal trimmable KV layer for the memory estimator (see
    test_prefix_cache_eviction.py for the shape rationale)."""

    class _FakeDtype:
        size = 4

    class _FakeArr:
        def __init__(self, n: int):
            self.shape = (n,)
            self.dtype = _FakeCacheLayer._FakeDtype()
            self.nbytes = n * 4

    def __init__(self, byte_size: int):
        n = max(1, byte_size // (2 * 4))
        self.state = (self._FakeArr(n), self._FakeArr(n))
        self.offset = n

    def is_trimmable(self) -> bool:
        return True


def _entry(byte_size: int = 1024):
    return [_FakeCacheLayer(byte_size)]


def _cache(max_mb: float = 8.0) -> MemoryAwarePrefixCache:
    return MemoryAwarePrefixCache(
        model=object(), config=MemoryCacheConfig(max_memory_mb=max_mb)
    )


def test_pin_existing_entry_survives_lru_eviction():
    cache = _cache(max_mb=8.0)
    pinned_tokens = list(range(64))
    cache.store(pinned_tokens, _entry(3 * 1024 * 1024))
    assert cache.pin_prefix(pinned_tokens) is True

    # Flood past the 8 MiB cap; LRU must reclaim the unpinned entries
    # and skip the pinned one even though it is the oldest.
    for i in range(1, 5):
        cache.store(list(range(i * 1000, i * 1000 + 64)), _entry(3 * 1024 * 1024))

    assert tuple(pinned_tokens) in cache._entries
    assert cache._entries[tuple(pinned_tokens)].protected is True
    assert cache.get_stats()["evictions"] >= 1


def test_pending_pin_applies_at_store():
    cache = _cache()
    tokens = list(range(64))
    # Pin BEFORE the entry exists — the real flow: the pin request lands
    # when the request is admitted, the boundary snapshot stores later.
    assert cache.pin_prefix(tokens) is False
    assert cache.store(tokens, _entry()) is True
    assert cache._entries[tuple(tokens)].protected is True
    assert not cache._pending_pins


def test_pinned_prefix_survives_prefix_subset_eviction():
    cache = _cache()
    prefix = list(range(64))
    cache.store(prefix, _entry())
    cache.pin_prefix(prefix)

    # Storing a strict extension with evict_prefixes=True normally
    # consumes the prefix entry; the pinned one must survive.
    cache.store(prefix + list(range(1000, 1064)), _entry(), evict_prefixes=True)
    assert tuple(prefix) in cache._entries


def test_all_protected_eviction_still_makes_progress():
    cache = _cache(max_mb=8.0)
    for i in range(2):
        tokens = list(range(i * 1000, i * 1000 + 64))
        cache.store(tokens, _entry(3 * 1024 * 1024))
        cache.pin_prefix(tokens)

    # Next store exceeds the cap with only protected entries present;
    # store() must not livelock — last resort evicts a pinned entry.
    assert cache.store(list(range(9000, 9064)), _entry(3 * 1024 * 1024)) is True
