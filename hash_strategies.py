"""
Lab 7: The Collision Resolver -- starter.

Complete the three classes below. See
Lab_07_The_Collision_Resolver.md, Part B, for the full requirements.
"""

from typing import Generic, Hashable, List, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_TOMBSTONE = object()  # sentinel marking a deleted open-addressing slot


class _ChainNode(Generic[K, V]):
    __slots__ = ("key", "value", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.next: Optional["_ChainNode[K, V]"] = None


class ChainedHashMap(Generic[K, V]):
    """Separate chaining: each bucket is a linked list of (key, value)."""

    def __init__(self, initial_size: int = 16) -> None:
        self._buckets: List[Optional[_ChainNode[K, V]]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def _resize(self, new_capacity: int) -> None:
        old_buckets = self._buckets
        self._buckets = [None] * new_capacity
        self._count = 0

        for head in old_buckets:
            curr = head
            while curr is not None:
                self.insert(curr.key, curr.value)
                curr = curr.next

    def insert(self, key: K, value: V) -> None:
        """Insert, or update in place if `key` already exists. Resize (double + rehash) once load factor > 0.75."""
        if (self._count + 1) / len(self._buckets) > 0.75:
            self._resize(len(self._buckets) * 2)

        idx = hash(key) % len(self._buckets)
        curr = self._buckets[idx]

        while curr is not None:
            if curr.key == key:
                curr.value = value
                return
            curr = curr.next

        new_node = _ChainNode(key, value)
        new_node.next = self._buckets[idx]
        self._buckets[idx] = new_node
        self._count += 1

    def get(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        idx = hash(key) % len(self._buckets)
        curr = self._buckets[idx]

        while curr is not None:
            if curr.key == key:
                return curr.value
            curr = curr.next

        raise KeyError(f"Key not found: {key}")

    def delete(self, key: K) -> None:
        """Remove `key`. Raise KeyError if missing."""
        idx = hash(key) % len(self._buckets)
        curr = self._buckets[idx]
        prev = None

        while curr is not None:
            if curr.key == key:
                if prev is None:
                    self._buckets[idx] = curr.next
                else:
                    prev.next = curr.next
                self._count -= 1
                return
            prev = curr
            curr = curr.next

        raise KeyError(f"Key not found: {key}")


class LinearProbingHashMap(Generic[K, V]):
    """Open addressing with linear probing and tombstone deletion."""

    def __init__(self, initial_size: int = 16) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def _resize(self, new_capacity: int) -> None:
        old_keys = self._keys
        old_values = self._values

        self._keys = [None] * new_capacity
        self._values = [None] * new_capacity
        self._count = 0

        for k, v in zip(old_keys, old_values):
            if k is not None and k is not _TOMBSTONE:
                self.insert(k, v)  # type: ignore

    def insert(self, key: K, value: V) -> None:
        """Resize (double + rehash) once load factor > 0.7."""
        if (self._count + 1) / len(self._keys) > 0.7:
            self._resize(len(self._keys) * 2)

        capacity = len(self._keys)
        start_idx = hash(key) % capacity
        first_tombstone_idx = None

        for i in range(capacity):
            idx = (start_idx + i) % capacity
            k = self._keys[idx]

            if k is None:
                target_idx = first_tombstone_idx if first_tombstone_idx is not None else idx
                self._keys[target_idx] = key
                self._values[target_idx] = value
                self._count += 1
                return
            elif k is _TOMBSTONE:
                if first_tombstone_idx is None:
                    first_tombstone_idx = idx
            elif k == key:
                self._values[idx] = value
                return

        if first_tombstone_idx is not None:
            self._keys[first_tombstone_idx] = key
            self._values[first_tombstone_idx] = value
            self._count += 1

    def search(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        capacity = len(self._keys)
        start_idx = hash(key) % capacity

        for i in range(capacity):
            idx = (start_idx + i) % capacity
            k = self._keys[idx]

            if k is None:
                raise KeyError(f"Key not found: {key}")
            if k is not _TOMBSTONE and k == key:
                return self._values[idx]  # type: ignore

        raise KeyError(f"Key not found: {key}")

    def delete(self, key: K) -> None:
        """Remove `key` using a tombstone (not None) so later probes don't stop early. Raise KeyError if missing."""
        capacity = len(self._keys)
        start_idx = hash(key) % capacity

        for i in range(capacity):
            idx = (start_idx + i) % capacity
            k = self._keys[idx]

            if k is None:
                raise KeyError(f"Key not found: {key}")
            if k is not _TOMBSTONE and k == key:
                self._keys[idx] = _TOMBSTONE
                self._values[idx] = None
                self._count -= 1
                return

        raise KeyError(f"Key not found: {key}")


class QuadraticProbingHashMap(Generic[K, V]):
    """
    Open addressing with quadratic probing and tombstone deletion.

    Pitfall to design around: with a power-of-2 table size, the probe
    sequence (idx + i^2) mod size does NOT reach every slot -- it can
    cycle through only about half of them, so the table can appear
    "full" and raise/loop forever even though empty slots exist
    elsewhere. Two standard fixes, pick one:
      (a) use a PRIME table size (so the quadratic sequence covers all
          slots whenever load factor < 1), or
      (b) resize proactively -- check load factor BEFORE attempting an
          insert's probe sequence, not only after a successful insert.
    Using both is safest.
    """

    def __init__(self, initial_size: int = 17) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def _next_prime(self, n: int) -> int:
        """Helper to compute the next prime number >= n."""
        def is_prime(num: int) -> bool:
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True

        prime = n
        while not is_prime(prime):
            prime += 1
        return prime

    def _resize(self, min_capacity: int) -> None:
        new_capacity = self._next_prime(min_capacity)
        old_keys = self._keys
        old_values = self._values

        self._keys = [None] * new_capacity
        self._values = [None] * new_capacity
        self._count = 0

        for k, v in zip(old_keys, old_values):
            if k is not None and k is not _TOMBSTONE:
                self.insert(k, v)  # type: ignore

    def insert(self, key: K, value: V) -> None:
        """Resize (grow + rehash) once load factor > 0.7 -- see the pitfall note above."""
        if (self._count + 1) / len(self._keys) > 0.7:
            self._resize(len(self._keys) * 2)

        capacity = len(self._keys)
        start_idx = hash(key) % capacity
        first_tombstone_idx = None

        for i in range(capacity):
            idx = (start_idx + i * i) % capacity
            k = self._keys[idx]

            if k is None:
                target_idx = first_tombstone_idx if first_tombstone_idx is not None else idx
                self._keys[target_idx] = key
                self._values[target_idx] = value
                self._count += 1
                return
            elif k is _TOMBSTONE:
                if first_tombstone_idx is None:
                    first_tombstone_idx = idx
            elif k == key:
                self._values[idx] = value
                return

        if first_tombstone_idx is not None:
            self._keys[first_tombstone_idx] = key
            self._values[first_tombstone_idx] = value
            self._count += 1

    def search(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        capacity = len(self._keys)
        start_idx = hash(key) % capacity

        for i in range(capacity):
            idx = (start_idx + i * i) % capacity
            k = self._keys[idx]

            if k is None:
                raise KeyError(f"Key not found: {key}")
            if k is not _TOMBSTONE and k == key:
                return self._values[idx]  # type: ignore

        raise KeyError(f"Key not found: {key}")

    def delete(self, key: K) -> None:
        """Remove `key` using a tombstone. Raise KeyError if missing."""
        capacity = len(self._keys)
        start_idx = hash(key) % capacity

        for i in range(capacity):
            idx = (start_idx + i * i) % capacity
            k = self._keys[idx]

            if k is None:
                raise KeyError(f"Key not found: {key}")
            if k is not _TOMBSTONE and k == key:
                self._keys[idx] = _TOMBSTONE
                self._values[idx] = None
                self._count -= 1
                return

        raise KeyError(f"Key not found: {key}")