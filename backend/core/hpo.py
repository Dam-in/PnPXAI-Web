"""Per-explainer hyperparameter optimization (HPO) for the /explain pipeline.

Tunes an explainer's hyperparameters together with its attribution postprocessor
(pooling + normalization) with Optuna, the same Objective the Optimizer page uses.
Image task only for now.

Measured HPO run times are stored in backend/precomputed/hpo_timing.json (written by
backend/scripts/precompute_samples.py) so the UI can tell a user roughly how long an
HPO run on their own image will take.
"""
import json
import os
import threading
from typing import Optional

PRECOMPUTED_DIR = os.path.join("backend", "precomputed")
HPO_TIMING_PATH = os.path.join(PRECOMPUTED_DIR, "hpo_timing.json")

HPO_METRIC = "AbPC"
HPO_N_TRIALS = 20
HPO_SEED = 42

# Used when no measured time exists for a (model, explainer) pair (e.g. a user-uploaded model)
_FALLBACK_SECONDS = 60.0

_timing_lock = threading.Lock()


def tune_explainer(explainer, model, inputs, targets,
                   metric_name: str = HPO_METRIC, n_trials: int = HPO_N_TRIALS,
                   should_stop=None):
    """Run Optuna over the explainer's and postprocessor's tunable params.

    `should_stop()` is checked after every trial (e.g. job cancellation).
    Returns (best_explainer, best_postprocessor, best_params, best_value).
    Raises if no trial completed.
    """
    import optuna
    from pnpxai.core.modality.modality import ImageModality
    from pnpxai.evaluator.optimizer.objectives import Objective
    from pnpxai.evaluator.optimizer.utils import load_sampler
    from backend.core.pipeline import _get_pnpxai_metric

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use ImageModality directly to avoid torch.fx tracing issues with some models
    modality = ImageModality()
    postprocessor = modality.get_default_postprocessors()[0]
    metric = _get_pnpxai_metric(metric_name, model, explainer)

    objective = Objective(
        explainer=explainer,
        postprocessor=postprocessor,
        metric=metric,
        modality=modality,
        inputs=inputs,
        targets=targets,
    )
    study = optuna.create_study(sampler=load_sampler("tpe", seed=HPO_SEED), direction="maximize")

    def _stop_cb(study, _trial):
        if should_stop is not None and should_stop():
            study.stop()

    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[_stop_cb])

    best = study.best_trial  # raises ValueError if every trial failed
    return (
        best.user_attrs["explainer"],
        best.user_attrs["postprocessor"],
        {k: _jsonable(v) for k, v in best.params.items()},
        float(best.value),
    )


def _jsonable(v):
    return v if isinstance(v, (str, int, float, bool)) or v is None else str(v)


def _load_timing() -> dict:
    try:
        with open(HPO_TIMING_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def record_timing(task: str, model_name: str, explainer_name: str, seconds: float) -> None:
    """Fold one measured HPO run time into the running average for (task, model, explainer)."""
    import fcntl
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    with _timing_lock, open(HPO_TIMING_PATH + ".lock", "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)  # several precompute processes may share the file
        timing = _load_timing()
        cur = timing.setdefault(task, {}).setdefault(model_name, {}).get(explainer_name)
        n = cur["n"] if cur else 0
        avg = cur["seconds"] if cur else 0.0
        timing[task][model_name][explainer_name] = {
            "seconds": round((avg * n + seconds) / (n + 1), 2),
            "n": n + 1,
        }
        tmp = HPO_TIMING_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
        os.replace(tmp, HPO_TIMING_PATH)


def estimate_seconds(task: str, model_name: str, explainer_names: list[str]) -> Optional[float]:
    """Expected wall time of an HPO explain run, from measured per-explainer times.

    Falls back to the same explainer's average across other models, then to a constant.
    Returns None when nothing has been measured for this task at all.
    """
    by_model = _load_timing().get(task, {})
    if not by_model:
        return None
    measured = by_model.get(model_name, {})
    total = 0.0
    for name in explainer_names:
        if name in measured:
            total += measured[name]["seconds"]
            continue
        others = [m[name]["seconds"] for m in by_model.values() if name in m]
        total += sum(others) / len(others) if others else _FALLBACK_SECONDS
    return total
