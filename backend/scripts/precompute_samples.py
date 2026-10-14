"""Precompute XAI results for every (built-in sample file x built-in model x
detected explainer) combination, so the live /explain endpoint can serve an
identical request from backend/precomputed/ instantly instead of recomputing.

Run from the project root (inside the container):
    python -m backend.scripts.precompute_samples [task ...] [--force]

With no arguments it precomputes **image and text only** — the time-series models have
no trained classifier yet, so caching their output isn't meaningful. Pass task names
explicitly to override:
    python -m backend.scripts.precompute_samples timeseries

Safe to re-run: a combination is skipped only if it is already cached AND still valid —
each entry records the CACHE_VERSION and a fingerprint of the model's weights, so
changing a model (retrain / re-architect / re-init) or bumping CACHE_VERSION makes stale
entries recompute automatically. `--force` recomputes regardless.

Image and text samples are precomputed with HPO (each explainer's hyperparameters tuned
by backend/core/hpo.py), since that's what the UI shows for its own samples. The
measured HPO times go to backend/precomputed/hpo_timing.json and back the run-time
estimate shown before an HPO run on a user's own image.

To split the work across GPUs, run one process per GPU with `--shard i/n`:
    CUDA_VISIBLE_DEVICES=2 python -m backend.scripts.precompute_samples image --shard 0/2
    CUDA_VISIBLE_DEVICES=3 python -m backend.scripts.precompute_samples image --shard 1/2

Requires sample_data/ to be present (it is git-ignored) and the models available.
"""
import fcntl
import glob
import json
import os
import shutil
import sys
import time

from backend.tasks import get_task_handler
from backend.core.job_manager import (
    create_job, store_uploaded_data, get_job, _remove_job, VISUALIZATION_DIR,
)
from backend.core.pipeline import run_explanation_pipeline
from backend.core.image_utils import load_and_validate_image
from backend.core.precompute_cache import (
    PRECOMPUTED_DIR, MANIFEST_PATH, file_sha256, sample_cache_dir,
    model_fingerprint, CACHE_VERSION,
)

SAMPLE_ROOT = "sample_data"
# Time-series is excluded by default: its models have no trained classifier yet
# (Simple-CNN/InceptionTime are fully random, MOMENT's classification head is random),
# so caching their output isn't meaningful. Re-enable once trained models land.
DEFAULT_TASKS = ["image", "text"]
ALL_TASKS = ["image", "text", "timeseries"]
# Tasks whose samples are served with HPO-tuned results
HPO_TASKS = {"image", "text"}


def _sample_files(task: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(SAMPLE_ROOT, task, "*")) if os.path.isfile(f))


def _load_data(task: str, raw: bytes):
    if task == "image":
        return load_and_validate_image(raw)
    if task == "text":
        return raw.decode("utf-8", errors="replace")
    return raw  # timeseries: raw CSV bytes


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    tmp_path = MANIFEST_PATH + f".tmp{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, MANIFEST_PATH)


def _store_entry(task: str, file_hash: str, model_name: str, entry: dict) -> None:
    """Write one entry into the on-disk manifest. Re-reads it under a file lock so
    parallel (sharded) precompute processes don't overwrite each other's entries."""
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    with open(MANIFEST_PATH + ".lock", "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        manifest = _load_manifest()
        manifest.setdefault(task, {}).setdefault(file_hash, {})[model_name] = entry
        _save_manifest(manifest)


def _result_to_cache_entry(r) -> dict:
    return {
        "explainer_name": r.explainer_name,
        "display_name": r.display_name,
        "status": r.status,
        "mu_fidelity": r.mu_fidelity,
        "abpc": r.abpc,
        "sensitivity": r.sensitivity,
        "complexity": r.complexity,
        "error_message": r.error_message,
        "token_attributions": [t.model_dump() for t in r.token_attributions] if r.token_attributions else None,
        "hpo_params": r.hpo_params,
        "hpo_seconds": r.hpo_seconds,
    }


def _record_hpo_timing(task: str, model_name: str, results, elapsed: float) -> None:
    """Per-explainer wall time = its own HPO time + an even share of the rest of the run
    (inference, attribution, metrics, rendering)."""
    from backend.core.hpo import record_timing
    done = [r for r in results if r.status == "completed" and r.hpo_seconds is not None]
    if not done:
        return
    overhead = max(elapsed - sum(r.hpo_seconds for r in done), 0.0) / len(done)
    for r in done:
        record_timing(task, model_name, r.explainer_name, r.hpo_seconds + overhead)


def precompute(tasks: list[str], force: bool = False, shard: tuple[int, int] = (0, 1)) -> None:
    manifest = _load_manifest()
    skipped, done, failed_combos = 0, 0, []
    combo_idx = -1

    for task in tasks:
        handler = get_task_handler(task)
        samples = _sample_files(task)
        print(f"\n=== {task}: {len(samples)} sample file(s) ===")

        for sample_path in samples:
            with open(sample_path, "rb") as f:
                raw = f.read()
            file_hash = file_sha256(raw)
            sample_name = os.path.basename(sample_path)

            for model_info in handler.get_models():
                model_name = model_info["name"]
                combo_idx += 1
                if combo_idx % shard[1] != shard[0]:
                    continue
                use_hpo = task in HPO_TASKS
                # Detection-driven list — same set the UI offers, so a cached hit is exact.
                explainer_names = [
                    e["name"] for e in handler.get_explainers(model_name)
                    if e.get("compatible", True)
                ]
                if not explainer_names:
                    continue

                # Fingerprint the actual weights so the entry self-invalidates if the
                # model changes later (retrained / re-architected / re-initialized).
                model_fp = model_fingerprint(handler.load_model(model_name))

                existing = manifest.get(task, {}).get(file_hash, {}).get(model_name)
                if existing and not force:
                    have = {r["explainer_name"] for r in existing.get("results", [])}
                    still_valid = (existing.get("cache_version") == CACHE_VERSION
                                   and existing.get("model_fp") == model_fp
                                   and bool(existing.get("hpo")) == use_hpo)
                    if still_valid and have >= set(explainer_names):
                        skipped += 1
                        continue

                mode = " with HPO" if use_hpo else ""
                print(f"  [{sample_name} + {model_name}] running {len(explainer_names)} explainer(s){mode}...", flush=True)
                t0 = time.time()

                data = _load_data(task, raw)
                job_id = f"precompute-{task}-{model_name}-{file_hash[:16]}"
                try:
                    store_uploaded_data(job_id, data, task)
                    create_job(job_id, task, model_name, explainer_names, "average")
                    run_explanation_pipeline(job_id, task, model_name, explainer_names, "average",
                                             {"use_hpo": use_hpo})
                    elapsed = time.time() - t0

                    job = get_job(job_id)
                    if job is None:
                        raise RuntimeError("job vanished after run_explanation_pipeline")

                    dest_dir = sample_cache_dir(task, model_name, file_hash)
                    os.makedirs(dest_dir, exist_ok=True)
                    src_dir = os.path.join(VISUALIZATION_DIR, job_id)
                    if os.path.isdir(src_dir):
                        for fname in os.listdir(src_dir):
                            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))

                    _store_entry(task, file_hash, model_name, {
                        "sample_name": sample_name,
                        "cache_version": CACHE_VERSION,
                        "model_fp": model_fp,
                        "hpo": use_hpo,
                        "predictions": [p.model_dump() for p in job.predictions] if job.predictions else None,
                        "results": [_result_to_cache_entry(r) for r in job.results],
                    })
                    if use_hpo:
                        _record_hpo_timing(task, model_name, job.results, elapsed)
                    done += 1
                    n_failed = sum(1 for r in job.results if r.status != "completed")
                    print(f"    done in {elapsed:.1f}s (status={job.status}, {n_failed} failed)", flush=True)
                    if n_failed:
                        for r in job.results:
                            if r.status != "completed":
                                print(f"      FAILED {r.explainer_name}: {r.error_message}")
                                failed_combos.append((task, sample_name, model_name, r.explainer_name, r.error_message))
                except Exception as e:
                    print(f"    ERROR: {type(e).__name__}: {e}")
                    failed_combos.append((task, sample_name, model_name, "ALL", str(e)))
                finally:
                    _remove_job(job_id)

    print(f"\n=== Done: {done} combo(s) computed, {skipped} already cached ===")
    if failed_combos:
        print(f"{len(failed_combos)} individual failure(s):")
        for task, sample_name, model_name, exp_name, err in failed_combos:
            print(f"  {task}/{sample_name}/{model_name}/{exp_name}: {err}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    shard = (0, 1)
    if "--shard" in argv:
        i = argv.index("--shard")
        k, n = argv[i + 1].split("/")
        shard = (int(k), int(n))
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    force = "--force" in argv
    for t in args:
        if t not in ALL_TASKS:
            sys.exit(f"unknown task: {t} (choose from {ALL_TASKS})")
    precompute(args or DEFAULT_TASKS, force=force, shard=shard)
