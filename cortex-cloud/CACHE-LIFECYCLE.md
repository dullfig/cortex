# Cache Lifecycle Semantics

From agentos-Claude, 2026-04-10. Defines the lifecycle for the cache_id-based
per-user KV cache system specified in api-spec.md.

## Ownership

- **agentos assigns cache_id** — cortex never invents one. Cortex receives
  cache_id as an opaque string routing key, never parses it.
- **Format**: `user-{user_id}` for production (e.g., `user-alice`,
  `user-12345`). Stable across sessions, pod restarts, account lifetime.
  One cache per user. Future: `user-alice:session-N` if session-scoped
  caches are ever needed.

## Source of truth

- **agentos's sled-backed KvCacheStore** is durable (survives restarts).
- **cortex's GPU memory** is a hot cache (volatile, can be lost anytime).
- Direction is asymmetric: cortex can lose everything and recover from sled.
  The reverse is not true.

## Lifecycle phases

### Cold start (first message ever from a user)

1. agentos sees `user-alice` has no entries in sled
2. agentos sends `POST /v1/chat/completions` with `cache_id: "user-alice"`
3. cortex returns 404 `cache_not_found`
4. agentos sends `POST /v1/cache/load` with empty entries (or skip if
   404 + empty sled is allowed — implementer's choice)
5. agentos retries `POST /v1/chat/completions`
6. cortex generates response, returns `new_cache_entries`
7. agentos sends `POST /v1/cache/append` AND writes entries to sled
8. `user-alice` now exists in both places

### Warm path (subsequent messages, same session)

1. agentos sends `POST /v1/chat/completions` with `cache_id: "user-alice"`
2. cortex uses resident cache, returns response + `new_cache_entries`
3. agentos appends to cortex (`POST /v1/cache/append`) AND to sled
4. ~0.5-2MB per turn

### Idle eviction (user inactive N minutes)

1. agentos calls `DELETE /v1/cache/user-alice`
2. cortex frees GPU memory
3. sled retains everything

### Reawaken after eviction

1. agentos sends `POST /v1/chat/completions` with `cache_id: "user-alice"`
2. cortex returns 404
3. agentos loads entries from sled (~42MB compressed for long history)
4. agentos sends `POST /v1/cache/load` with full entries array
5. agentos retries chat completion
6. Warm path from here

### Cortex restart / pod death

- All cache_ids return 404
- First message from each user triggers sled→cortex reload
- Recovery is per-user lazy, not eager

### Cortex auto-evicts under GPU memory pressure (LRU)

- Returns 404 next time agentos uses that cache_id
- agentos handles identically to idle eviction recovery
- **The 404 is the protocol** — no separate notification needed

## Concurrency

- agentos enforces a **per-user mutex** on the request side
- Two simultaneous messages from the same user serialize at agentos
- Cortex doesn't need to worry about concurrent appends to the same cache
- Different cache_ids are fully independent

## What cortex owns

- The cache pool (`HashMap<String, ModelKvCache>` in GPU memory)
- LRU eviction policy under memory pressure
- Returning 404 promptly when a cache_id isn't resident
- Semantics: "load replaces, append extends"

## What cortex does NOT own

- User identity, permissions, billing
- Cache retention policy (agentos drives DELETEs explicitly)
- Durability (agentos's sled handles persistence)
- Cross-cache concurrency (every cache_id is independent)
