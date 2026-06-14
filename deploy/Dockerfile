# cortex-server deploy image — Phase Q "cortex helper".
#
# Multistage: build the Rust binary, then ship a slim runtime with the
# Vulkan loader + the onnxruntime shared lib that `ort` (shim feature)
# links. cortex is wgpu->Vulkan, NOT CUDA — the NVIDIA Vulkan ICD is
# injected from the host at run time by the NVIDIA Container Toolkit,
# so it is NOT installed here. You MUST run with the `graphics`
# capability (see deploy/README.md + docker-compose.yml), or Vulkan
# finds no device and cortex aborts with "no GPU adapter".
#
# Image is MODEL-AGNOSTIC: provide the GGUF via a volume mount at
# /models (default) or set CORTEX_MODEL_URL to fetch at boot. Nothing
# model-specific is baked.

# ---- build stage --------------------------------------------------------
FROM rust:1-bookworm AS build
WORKDIR /src

# `ort` downloads onnxruntime binaries at build time (download-binaries
# feature) and crates.io is needed — this stage requires network.
# Copy the whole workspace (cortex / cortex-cloud / cortex-local).
# pinky/ is excluded from the workspace and not needed for the server.
COPY Cargo.toml Cargo.lock ./
COPY cortex/ cortex/
COPY cortex-cloud/ cortex-cloud/
COPY cortex-local/ cortex-local/

# Build only the server binary, release.
RUN cargo build --release -p cortex-cloud --bin cortex-server

# Locate the onnxruntime shared lib that `ort` fetched/built so we can
# carry it into the runtime stage. download-binaries places it under the
# target dir; collect any libonnxruntime*.so into a stable spot.
RUN set -eux; \
    mkdir -p /artifacts/lib; \
    cp target/release/cortex-server /artifacts/cortex-server; \
    find target/release -name 'libonnxruntime*.so*' -exec cp -av {} /artifacts/lib/ \; ; \
    ls -l /artifacts /artifacts/lib

# ---- runtime stage ------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# libvulkan1 = the Vulkan loader (the loader is needed in-container; the
# NVIDIA ICD itself is injected by the container toolkit at run time).
# ca-certificates + curl for the optional CORTEX_MODEL_URL fetch.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libvulkan1 ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*

# Binary + the onnxruntime lib it links.
COPY --from=build /artifacts/cortex-server /usr/local/bin/cortex-server
COPY --from=build /artifacts/lib/ /usr/local/lib/cortex/
ENV LD_LIBRARY_PATH=/usr/local/lib/cortex:${LD_LIBRARY_PATH}

# Vulkan: make the toolkit inject the NVIDIA ICD + libs. These are also
# set in docker-compose.yml; duplicated here so a bare `docker run
# --gpus all` works too. `graphics` is the load-bearing capability.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Model lands here (volume mount, or fetched by the entrypoint).
RUN mkdir -p /models
VOLUME ["/models"]

# cortex listens here; ONLY the Caddy sidecar should reach it (compose
# keeps it off published ports). Documented, not published by the image.
EXPOSE 8100

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
