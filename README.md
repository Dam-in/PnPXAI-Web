# XAI Demo Platform

<img width="563" height="163" alt="image" src="https://github.com/user-attachments/assets/e07ecafa-5d7d-4827-ba5b-fc015625f3f7" />


An interactive **Explainable AI (XAI)** demo platform supporting **Image Classification**, **Text Classification**, and **Time-Series Forecasting**.

The platform enables users to upload data, select pre-trained models and XAI explainers, visualize attribution maps, and compare explanation quality using multiple evaluation metrics.

---

## Features

### Supported Tasks

### Image Classification
- ResNet-50
- VGG-16
- DenseNet-121

All three are torchvision ImageNet-pretrained models.

### Text Classification
- BERT (ToxiGen) — less toxic / ambiguous / toxic

Fine-tuned locally on ToxiGen (ACL 2022) with `backend/scripts/train_toxigen.py`. The
three classes are the paper's own harm bands (<3, =3, >3 on the 1–5 annotator scale).

### Time-Series Forecasting
- iTransformer
- TimesNet

Each model has one checkpoint per sample dataset (ETTh1, illness); choosing the dataset
selects the checkpoint trained on it. iTransformer runs on any channel count, while a
TimesNet checkpoint only runs on the channel count it was trained with.

### Custom Models
Besides the presets, every task accepts a HuggingFace model ID or an uploaded weights
file (up to 500MB, kept for 6 hours). The explainer list is then detected for that model.

---

## XAI Explainers

The available explainers are automatically determined for each model using **PnPXAI's architecture detection**, ensuring that only compatible explainers are presented.

| Task | Model | Explainers |
|---|---|---|
| Image | ResNet-50, VGG-16 | 11 — Gradient, Gradient×Input, SmoothGrad, VarGrad, IntegratedGradients, GradCAM, GuidedGradCAM, LRP (Uniform Epsilon), RAP, LIME, KernelSHAP |
| Image | DenseNet-121 | 9 — the above minus LRP and RAP |
| Text | BERT (ToxiGen) | 8 — gradient family + LRP (Uniform Epsilon) + LIME + KernelSHAP |
| Time-Series | iTransformer | 8 — gradient family + AttentionRollout + TransformerAttribution + LRP (Uniform Epsilon) |
| Time-Series | TimesNet | 9 — gradient family + LRP ×4 (Uniform Epsilon, Epsilon+, Epsilon-γ-box, α2β1) |

"Gradient family" = Gradient, Gradient×Input, SmoothGrad, VarGrad, IntegratedGradients.

Detection can over-report: it recommends methods whose assumptions don't hold for the
concrete model. Those are filtered out per model/modality, verified by actually running
every sample × model × explainer combination:

- **Concatenating architectures break zennit's LRP** — DenseNet's dense blocks hit
  `size of tensor a (32) must match tensor b (96)`.
- **RAP** has no rule for `adaptive_avg_pool2d` (DenseNet) and fails on time-series.
- **CAM** assumes 2D spatial feature maps, so it is excluded for time-series.
- **LIME / KernelSHAP** fail on time-series inputs.
- **Image shows one LRP variant only** — PnPXAI also recommends Epsilon+, Epsilon-γ-box
  and α2β1, but only Uniform Epsilon is exposed for image models.

Text uses the detected list as-is, with nothing filtered out.

> Note: the time-series models are **forecasters**, not classifiers. The explained output is
> the next predicted step of the target series (the last CSV column), so the attribution
> shows which past timesteps drove that prediction.

### Evaluation Metrics
| Task | Metrics |
|---|---|
| Image | MuFidelity, AbPC, Sensitivity, Complexity |
| Text | AbPC, Sensitivity, Complexity |
| Time-Series | Sensitivity, Complexity |

- MuFidelity and AbPC are classification-only, so they are skipped for forecasters.
- Results ranked by selected metric or average score

### Hyperparameter Optimization (HPO)
- Image and text only: optionally tune each explainer's hyperparameters and attribution
  postprocessing (pooling + normalization) with Optuna before explaining
  (AbPC, 20 trials, seed 42)
- The UI shows an estimated run time first, from measured times in
  `backend/precomputed/hpo_timing.json`

### Visualization
- **Image**: Attribution heatmaps with side-by-side comparison
- **Text**: Token-level attribution highlighting + bar chart
- **Time-Series**: White→red background attribution (magnitude) + signal overlay
  - Input & forecast chart (context window + predicted horizon, chained back-test windows)
  - Multi-variate support (top variables ranked by importance)
  - Time axis labels from a date/TIME column
  - Expanded view (5×3 grid) + ZIP bundle download (Excel + per-variable PNGs)

### Data Handling
- Sample data with dataset info & source links
  - Image: `bird.jpg`, `cat.jpg`, `goldfish.JPEG`
  - Text: five ToxiGen statements, one per annotator harm level (`toxigen_l1` … `toxigen_l5`)
  - Time-Series: `ETTh1.csv`, `illness.csv`
- Drag-and-drop upload (images up to 10MB, other data files up to 100MB)
- Auto-detection of non-series columns (date/TIME → time axis, ID/index columns dropped)

### Optimizer Page
- Hyperparameter optimization for XAI explainers
- History tracking with result comparison (last 5 runs)

## Tech Stack

- **Frontend**: Next.js 14 + React 18 + TypeScript + Tailwind CSS (runs on the host, Node 20)
- **Backend**: FastAPI + PyTorch + PnPXAI (runs inside a GPU Docker container)
- **Time-Series Models**: iTransformer, TimesNet (checkpoints trained per dataset)
- **Metrics**: PnPXAI evaluator (MuFidelity, AbPC, Sensitivity, Complexity)
- **HPO**: Optuna (via PnPXAI)

## Operating Model

The **backend runs inside a GPU Docker container** — all Python dependencies
(torch, transformers, pnpxai, …) live in the container image, not on
the host. The **frontend runs on the host** with Node and proxies API calls to
the backend. Model weights are cached in a shared, mounted `models/` volume so
they survive container recreation and are shared across containers.

```
Host                                   Docker container (e.g. haeun_pnp)
────────────────────────────────      ─────────────────────────────────────────
frontend (Next.js) :3000  ──/api──►    backend (FastAPI/uvicorn) :8000
  repo mounted at ./                   repo mounted at /project (live --reload)
  BACKEND_URL → :8001                  /project/models  ← shared model volume
                                       GPU via /dev/nvidia* device mounts
```

Ports: **frontend `:3000`**, **backend `:8001` on host → `:8000` in container**.
(The host port is configurable via `-p`; the frontend backend URL is set with
`BACKEND_URL`, default `http://localhost:8000`.)

## Quick Start

### 1. Backend (Docker container)

This host's Docker has no NVIDIA runtime (`--gpus all` fails), so GPUs are passed
in by bind-mounting the device nodes and driver libraries. Create the container
(adjust name, host port, and mount paths for your setup):

```bash
docker run -d -it --name haeun_pnp --ipc=host --restart unless-stopped \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8001:8000 \
  --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia0 \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  -v $(pwd):/project \
  -v /data8/haeun/PnPXAI/models:/project/models \
  -v /data8/haeun/PnPXAI/sample_data:/project/sample_data \
  nvcr.io/nvidia/pytorch:23.07-py3
# NOTE: add every /dev/nvidiaN and libnvidia-*.so / libcuda.so* your host exposes.
```

Install Python dependencies inside the container (pinned versions; `setuptools`
must stay `<81` so `pkg_resources` survives):

```bash
docker exec haeun_pnp pip install \
  torch==2.13.0 torchvision==0.28.0 \
  transformers==5.14.0 huggingface_hub==1.23.0 tokenizers==0.22.2 safetensors==0.8.0 \
  pnpxai==0.1.4 captum==0.9.0 \
  fastapi==0.139.0 uvicorn==0.51.0 python-multipart==0.0.32 matplotlib==3.7.2 openpyxl \
  soxr==1.1.0 pandas==2.3.3 setuptools==80.10.2
```

Populate the shared `models/` volume (one time; see [Model Storage](#model-storage)):

```bash
# image models (torchvision ImageNet weights)
docker exec -w /project haeun_pnp python -m backend.scripts.download_models --only image
# text model — fine-tuned on ToxiGen here; it does not exist on any hub
docker exec -w /project haeun_pnp python -m backend.scripts.train_toxigen
# time-series checkpoints are not downloaded: place the trained .pth files in models/timeseries/
```

Run the backend.

```bash
docker exec -d -w /project haeun_pnp \
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at http://localhost:8001

### 2. Frontend (host)

```bash
cd frontend
npm install
echo "BACKEND_URL=http://localhost:8001" > .env.local   # point at your backend port
npm run dev
```

App runs at http://localhost:3000

### Everyday use / after a reboot

The container uses `--restart unless-stopped`, so it comes back automatically
after a host reboot — but the uvicorn process (started via `docker exec`) does
not, so restart it:

```bash
docker start haeun_pnp                                   # if stopped
docker exec -d -w /project haeun_pnp \
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
docker exec haeun_pnp tail -f /var/log/uvicorn_haeun.log # tail logs (if redirected there)
```

Editing code on the host is reflected live (`--reload`) — no container restart
needed. You only need the container running to **run/test** the backend, not to
edit code.

## Precomputed Sample Results (optional)

XAI results for the built-in sample data can be precomputed so `/explain` serves an
identical request instantly instead of recomputing:

```bash
docker exec -w /project haeun_pnp python -m backend.scripts.precompute_samples [task ...] [--force]
```

- With no task names it precomputes **image and text only**; pass `timeseries` to include it.
- Image and text samples are precomputed **with HPO**, and the measured HPO times are
  written to `backend/precomputed/hpo_timing.json` (used for the HPO time estimate).
- Split across GPUs with one process per GPU: `CUDA_VISIBLE_DEVICES=2 … --shard 0/2`.

- Cache lives in `backend/precomputed/` and is **git-ignored** — regenerate after cloning.
- Matched by **file SHA-256 + model name**, so it only hits when the uploaded file is
  byte-identical to a sample. Everything else computes normally.
- **Self-invalidating**: each entry records `CACHE_VERSION` and a fingerprint of the
  model's weights. Swapping in a retrained/re-architected model makes stale entries
  recompute automatically — no manual clearing. Bump `CACHE_VERSION` in
  `backend/core/precompute_cache.py` when pipeline/rendering logic changes (weights are
  unchanged then, so the fingerprint can't catch it).
- Cached hits are revealed one explainer at a time (2s each) so progress still shows;
  set `PNPXAI_PRECOMPUTE_STEP_DELAY=0` to serve instantly.

## Compute Device (GPU)

Models and all inference/attribution tensors run on GPU when one is available.

- Resolution order (`backend/core/device.py`): **`$PNPXAI_DEVICE`** (e.g. `cuda`, `cuda:1`,
  `cpu`) → else `cuda` if available → else `cpu`.
- Models are moved to the device on load; the pipeline moves inputs/targets to the
  *model's own* device, so everything stays consistent.
- Force CPU with `PNPXAI_DEVICE=cpu` (useful for debugging).

### VRAM release when idle

Models are **offloaded back to CPU as soon as the last in-flight job finishes**, so an
idle server holds no model VRAM. A job counter guards this — concurrent jobs never have
their models pulled out from under them.

The model objects stay cached in the task handlers, so the next job only pays a
host→device copy (no re-download or re-instantiation). Measured on VGG-16:

| | torch allocated | model device |
|---|---|---|
| after GPU load | 554.7 MB | `cuda:0` |
| after offload | **0 MB** | `cpu` |

A residual ~160–230 MiB stays in `nvidia-smi` while the server runs — that's PyTorch's
**CUDA context**, which cannot be freed without exiting the process. All model memory is
returned.

Set `PNPXAI_KEEP_ON_GPU=1` to disable offloading (lowest latency, holds VRAM).

Verify it's actually on GPU:
```bash
docker exec <container> nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

> GPU matters a lot here: Integrated Gradients on ResNet-50 measured **11.4s on CPU vs
> 0.067s on GPU (~170×)**. Note the container needs the GPU device/library mounts (see
> Quick Start) — without them `torch.cuda.is_available()` is False and it silently
> falls back to CPU.

## Model Storage

Model weights are cached locally instead of relying on the ephemeral
`~/.cache` inside the container.

- **Location**: `$PNPXAI_MODEL_DIR`, else `<repo>/models` — which resolves to the
  mounted `/project/models` volume in the container (shared across containers,
  survives recreation). Resolved by `backend/core/model_paths.py`.
- **Layout**:

  ```
  models/
    image/resnet50.pth, vgg16.pth, densenet121.pth   (torchvision state_dicts)
    text/toxigen-bert/                               (transformers save_pretrained)
    timeseries/iTransformer_etth1.pth, iTransformer_illness.pth,
               timesnet_etth1.pth, timesnet_illness.pth
  ```

- **Populate**:
  - Image: `python -m backend.scripts.download_models --only image`
  - Text: `python -m backend.scripts.train_toxigen` — the model is fine-tuned here and
    exists on no hub. (`download_models --only text` fetches `hatexplain-bert`, a retired
    preset kept only as the baseline for `backend/scripts/eval_toxigen.py`.)
  - Time-Series: trained checkpoints are placed in `models/timeseries/` by hand.
- **Loading**: handlers load from this directory with `local_files_only=True`
  (offline-capable). Image models **fall back to torchvision** when absent; the text and
  time-series presets have no remote fallback and fail with a message naming the
  missing path.
- Weights are git-ignored (`models/.gitignore`); repopulate on a new deployment.

## Architecture

```
User
 │
 ├── Upload Data
 │
 ├── Select Model
 │
 ├── Select Explainers
 │
 ▼
Frontend (Next.js)
 │
 ▼
FastAPI Backend
 │
 ├── Load Model
 ├── Prediction
 ├── Attribution
 ├── Evaluation Metrics
 └── Visualization
 │
 ▼
Frontend Visualization
```

---

# REST API

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/tasks` | Supported tasks |
| GET | `/api/models` | Supported models |
| GET | `/api/explainers` | Supported explainers (detected per model) |
| GET | `/api/recommend` | Detected architectures + recommended explainers |
| GET | `/api/samples/{task}` | Sample data list |
| GET | `/api/samples/{task}/{filename}` | Sample data file |
| GET | `/api/validate-model` | Validate a HuggingFace model ID (or checkpoint URL for time-series) |
| POST | `/api/upload-model` | Upload model weights |
| GET | `/api/hpo/estimate` | Estimated HPO run time |
| POST | `/api/explain` | Run explanation (async job) |
| GET | `/api/jobs/{id}` | Job status |
| POST | `/api/jobs/{id}/cancel` | Cancel a job |
| GET | `/api/jobs/{id}/visualizations/{name}` | Visualization |
| GET | `/api/jobs/{id}/original/{name}` | Original input data (image / text / CSV) |
| POST | `/api/detect-rank` | Start explainer detection + ranking job |
| GET | `/api/detect-rank/{id}` | Detection + ranking job status |
| GET | `/api/optimizer/params/{explainer}` | Tunable parameters of an explainer |
| POST | `/api/optimizer/optimize` | Hyperparameter optimization |
| POST | `/api/optimizer/custom` | Run with custom parameters |
| GET | `/api/optimizer/history` | Optimizer history |
| GET / DELETE | `/api/optimizer/history/{id}` | Get / delete a history record |
| POST | `/api/optimizer/history/{id}/custom` | Re-run a history record with custom parameters |

---

# Project Structure

```
backend/
    api/          REST routes + schemas
    core/         pipeline, explainer detection, HPO, jobs, precompute cache
    models/       iTransformer / TimesNet implementations
    tasks/        per-task handlers (image, text, timeseries)
    renderers/    heatmap / token / time-series rendering
    optimizer/    Optimizer page service
    scripts/      model download, ToxiGen training, sample precompute
    tests/

frontend/
    src/app/         pages (main, /optimizer)
    src/components/
    src/hooks/
    src/lib/         API client, types
    public/

models/

sample_data/
```

---

# Notes

- The backend runs in the container only — the host has no torch/transformers, so
  code can be edited on the host but must be **run/tested inside the container**.
- Keep `setuptools<81` in the container: newer setuptools removes `pkg_resources`,
  which breaks the librosa → transformers audio import chain.
- Files written into mounted volumes by the container are owned by `root`; fix with
  `docker exec <container> chown -R <uid>:<gid> /project/models` if the host user
  needs to manage them.
- LIME and KernelSHAP are slower (~20-30s) due to the perturbation-based approach.
- For large time-series (>512 timesteps), sliding window attribution is used automatically.
- LRP explainers use deep-copied models to prevent state corruption.
