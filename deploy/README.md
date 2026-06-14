# cortex deploy helper

Box-local packaging to run a **secure** cortex-server on a remote GPU
box. cortex stays inference-only; this directory is everything needed to
run it behind TLS + auth. It does **not** provision boxes — your
orchestrator (AgentOS via the rental provider's API) pulls the published
image and launches the stack; clients then talk HTTPS.

```
registry(cortex img) --pull--> GPU box:  Caddy :443  (TLS + API key)
                                            └─► cortex-server :8100 (private)
                                                model: volume or fetch-at-boot
AgentOS --HTTPS + Bearer key--> Caddy
AgentOS --provider API--------> launch / stop the box
```

## What's here

| file | role |
|---|---|
| `Dockerfile` | multistage build of `cortex-server`; slim Vulkan+onnxruntime runtime |
| `entrypoint.sh` | resolves the model (mount or `CORTEX_MODEL_URL`), exec's the server with env-configured flags |
| `docker-compose.yml` | `cortex` (private) + `caddy` (public :443); GPU wired |
| `Caddyfile` | TLS termination + Bearer API-key gate (`/health` open) |
| `.env.example` | copy to `.env`, fill secrets |

## ⚠️ The one gotcha: Vulkan needs the `graphics` capability

cortex is **wgpu → Vulkan**, not CUDA. The NVIDIA Container Toolkit
injects the NVIDIA Vulkan ICD into the container **only when the
`graphics` driver capability is enabled**. Most GPU-container guides
enable `compute,utility` only — with those, wgpu finds **no device** and
cortex aborts. The compose file sets:

```
NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
NVIDIA_VISIBLE_DEVICES=all
```

Confirm success in the logs: `GPU adapter selected … backend=Vulkan`.
If you see "no GPU adapter," the capability/ICD injection is the cause —
fallbacks: an `nvidia/cuda` base image, or installing the ICD JSON
explicitly.

## Run it (on a Linux NVIDIA box, toolkit installed)

```bash
cp deploy/.env.example deploy/.env
# edit deploy/.env: set CORTEX_API_KEY (long random), CORTEX_DOMAIN
# (or leave blank for bare-IP), and model delivery.

# from the repo root (build context is the root):
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
```

## Model delivery (image is model-agnostic)

- **Mount** (default): put `model.gguf` in `CORTEX_MODELS_DIR` (default
  `./models`), mounted to `/models`.
- **Fetch at boot**: set `CORTEX_MODEL_URL` to a GGUF URL; the entrypoint
  downloads it on first start (good for ephemeral instances + object
  storage). One small image serves any model.
- *(Alternative, not default)* bake the GGUF into the image for a fully
  self-contained artifact — simplest to launch, but a multi-GB,
  per-model image.

## TLS

- **Domain** (`CORTEX_DOMAIN=cortex.example.com`): Caddy gets a real
  Let's Encrypt cert automatically (needs ports 80+443 reachable, DNS
  pointed at the box).
- **Bare IP** (`CORTEX_DOMAIN` blank → address `:443`): **uncomment the
  `tls internal` line in the `Caddyfile`** so Caddy serves its own CA
  cert (it can't get a public cert without a hostname). AgentOS must then
  trust/pin that cert, or run inside a private network/VPN. Do *not*
  enable `tls internal` with a real domain — it disables Let's Encrypt.

## Auth

Every request except `/health` needs `Authorization: Bearer
$CORTEX_API_KEY`. Set the key in `.env` (never commit it). cortex has no
auth of its own by design — Caddy is the only door, and cortex never
publishes a host port.

```bash
curl -H "Authorization: Bearer $CORTEX_API_KEY" https://HOST/health
curl -H "Authorization: Bearer $CORTEX_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"model":"cortex","messages":[{"role":"user","content":"hi"}]}' \
     https://HOST/v1/chat/completions
```

## AgentOS integration

Point AgentOS's `OpenAiClient` at `https://HOST/v1` with the Bearer key.
For elastic spin-up, AgentOS calls the **provider's** API to launch a box
that pulls `cortex-server:latest` and runs this compose stack — no cortex
control-plane endpoint involved. Keep session affinity to one instance
while a retrieval cache is loaded (the cache is server-side state).

## Notes / caveats

- **onnxruntime**: `ort` (shim feature) links libonnxruntime; the
  Dockerfile copies it from the build stage into `/usr/local/lib/cortex`
  with `LD_LIBRARY_PATH` set. If a future build relocates it, adjust the
  `find` in the build stage.
- **Cold start** = image pull + (optional) model fetch + wgpu init.
  Fine for warm instances; budget it for ephemeral launch.
- **Not in scope here**: provider provisioning/autoscaling, multi-instance
  shared cache / session affinity, k8s manifests.
- **Validation status**: the GPU/Vulkan-in-container path is best-effort
  and must be confirmed on a real Linux NVIDIA box (`GPU adapter
  selected … Vulkan` in logs). Structural bits (compose/Caddyfile) are
  validatable anywhere.
```
