#!/usr/bin/env bash
# cortex-server container entrypoint (Phase Q).
#
# Resolves the model (volume-mounted, or fetched from CORTEX_MODEL_URL),
# then exec's cortex-server with flags from environment variables. Keeps
# the image model-agnostic and config-via-env so one image serves any
# model on any box.
#
# Env:
#   CORTEX_MODEL          path to GGUF inside the container
#                         (default /models/model.gguf)
#   CORTEX_MODEL_URL      if set and the model file is absent, fetch it here
#   CORTEX_PORT           listen port (default 8100; compose proxies to it)
#   CORTEX_BIND           bind addr (default 0.0.0.0 — REQUIRED for the
#                         Caddy sidecar to reach it over the compose net;
#                         isolation comes from NOT publishing this port to
#                         the host, never from a 127.0.0.1 bind)
#   CORTEX_MAX_SEQ        --max-seq-len (default 4096)
#   CORTEX_ENABLE_CACHE / CORTEX_ENABLE_RETRIEVE /
#   CORTEX_ENABLE_POLAR_CACHE / CORTEX_ENABLE_SHIMS
#                         set to "1"/"true" to add the corresponding flag
#   CORTEX_QJL_PROJECTIONS / CORTEX_QJL_SEED   passthrough if set
#   CORTEX_EXTRA_ARGS     raw extra args appended verbatim (escape hatch)
set -euo pipefail

MODEL="${CORTEX_MODEL:-/models/model.gguf}"

if [[ ! -f "$MODEL" ]]; then
    if [[ -n "${CORTEX_MODEL_URL:-}" ]]; then
        echo "entrypoint: fetching model from \$CORTEX_MODEL_URL -> $MODEL"
        mkdir -p "$(dirname "$MODEL")"
        # -f fail on HTTP error, -L follow redirects, -S show errors
        curl -fSL --retry 3 -o "$MODEL.partial" "$CORTEX_MODEL_URL"
        mv "$MODEL.partial" "$MODEL"
    else
        echo "entrypoint: ERROR no model at $MODEL and CORTEX_MODEL_URL unset." >&2
        echo "  Mount a GGUF at /models or set CORTEX_MODEL_URL." >&2
        exit 1
    fi
fi

args=(--model "$MODEL"
      --port "${CORTEX_PORT:-8100}"
      --bind "${CORTEX_BIND:-0.0.0.0}"
      --max-seq-len "${CORTEX_MAX_SEQ:-4096}")

is_on() { case "${1:-}" in 1|true|TRUE|yes|on) return 0;; *) return 1;; esac; }
is_on "${CORTEX_ENABLE_CACHE:-}"        && args+=(--enable-cache)
is_on "${CORTEX_ENABLE_RETRIEVE:-}"     && args+=(--enable-retrieve)
is_on "${CORTEX_ENABLE_POLAR_CACHE:-}"  && args+=(--enable-polar-cache)
is_on "${CORTEX_ENABLE_SHIMS:-}"        && args+=(--enable-shims)
[[ -n "${CORTEX_QJL_PROJECTIONS:-}" ]]  && args+=(--qjl-projections "$CORTEX_QJL_PROJECTIONS")
[[ -n "${CORTEX_QJL_SEED:-}" ]]         && args+=(--qjl-seed "$CORTEX_QJL_SEED")
# shellcheck disable=SC2206
[[ -n "${CORTEX_EXTRA_ARGS:-}" ]]       && args+=(${CORTEX_EXTRA_ARGS})

echo "entrypoint: exec cortex-server ${args[*]}"
exec cortex-server "${args[@]}"
