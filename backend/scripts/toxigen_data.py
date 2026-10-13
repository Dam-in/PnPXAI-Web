"""Load the ToxiGen `annotated` split for training and evaluating the text model.

The dataset is fetched as parquet straight from the HuggingFace CDN rather than via
the `datasets` library: `datasets` is not installed in the container and pulling it in
would drag pyarrow/fsspec/huggingface_hub versions along with it, which the pinned
transformers 5.14.0 / torch 2.13.0 environment cannot absorb. The two files are ~770KB
combined, so a plain download + `pandas.read_parquet` is cheaper than the dependency.

Files are cached under sample_data/.cache/toxigen/ so re-runs work offline. That
directory is invisible to precompute_samples.py, which only globs sample_data/<task>/*.

    from backend.scripts.toxigen_data import load_annotated, to_label
    train_df, test_df = load_annotated()

Self-test:
    python -m backend.scripts.toxigen_data --selftest
"""
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

_BASE = ("https://huggingface.co/datasets/toxigen/toxigen-data/resolve/"
         "refs%2Fconvert%2Fparquet/annotated")
_SPLITS = {"train": f"{_BASE}/train/0000.parquet", "test": f"{_BASE}/test/0000.parquet"}

CACHE_DIR = os.path.join("sample_data", ".cache", "toxigen")

# The paper's three bands (footnote 8): "scores <3: 'non-toxic', =3: 'ambiguous',
# >3: 'toxic'". We call the first one "less toxic"; see backend/tasks/text.py.
NUM_CLASSES = 3
LESS_TOXIC, AMBIGUOUS, TOXIC = 0, 1, 2

# The 13 canonical target groups, as they appear in the train split.
CANONICAL_GROUPS = [
    "asian", "black", "chinese", "jewish", "latino", "lgbtq", "mental_dis",
    "mexican", "middle_east", "muslim", "native_american", "physical_dis", "women",
]

# Race/ethnicity groups — used only to pick demo samples, never to filter training data.
CLASS_NAMES = {LESS_TOXIC: "less toxic", AMBIGUOUS: "ambiguous", TOXIC: "toxic"}

RACE_GROUPS = [
    "black", "asian", "chinese", "latino", "mexican", "native_american", "middle_east",
]

# The 940-row test split labels the same 13 groups with free-text phrases instead of the
# canonical slugs the train split uses. Filtering or grouping without folding these in
# silently drops the entire test split, so normalize_target_group() raises on anything
# it cannot map rather than letting the rows disappear.
_TARGET_GROUP_ALIASES = {
    "asian folks": "asian",
    "black folks / african-americans": "black",
    "black/african-american folks": "black",
    "chinese folks": "chinese",
    "jewish folks": "jewish",
    "latino/hispanic folks": "latino",
    "lgbtq+ folks": "lgbtq",
    "folks with mental disabilities": "mental_dis",
    "mexican folks": "mexican",
    "middle eastern folks": "middle_east",
    "muslim folks": "muslim",
    "native american folks": "native_american",
    "native american/indigenous folks": "native_american",
    "folks with physical disabilities": "physical_dis",
    "women": "women",
}


def _download(split: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, f"{split}.parquet")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = _SPLITS[split]
    print(f"[toxigen] downloading {split} <- {url}")
    tmp = dest + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"could not fetch ToxiGen {split} parquet from {url}: {type(e).__name__}: {e}\n"
            f"If the HF parquet URL scheme changed, download it by hand into {cache_dir}/"
        ) from e
    print(f"[toxigen] cached -> {dest} ({os.path.getsize(dest)/1e3:.0f} KB)")
    return dest


def normalize_target_group(df: pd.DataFrame) -> pd.DataFrame:
    """Fold the test split's free-text group phrases onto the canonical slugs.

    Raises if a value maps to neither, so a future dataset revision surfaces as a loud
    failure instead of a quietly shrunken dataset.
    """
    df = df.copy()
    df["target_group"] = df["target_group"].str.strip().str.lower()
    known = set(CANONICAL_GROUPS)
    df["target_group"] = df["target_group"].map(
        lambda g: g if g in known else _TARGET_GROUP_ALIASES.get(g, None)
    )
    unmapped = df["target_group"].isna()
    if unmapped.any():
        raise ValueError(
            f"{int(unmapped.sum())} row(s) have a target_group that is neither canonical "
            f"nor a known alias. Add them to _TARGET_GROUP_ALIASES."
        )
    return df


def paper_score(toxicity_ai, toxicity_human) -> float:
    """The score footnote 8 bins: max(HARMFULIFAI, HARMFULIFHUMAN), each a mean of 3
    annotators on the 1-5 harm scale, so it moves in thirds (2.67, 3.0, 3.33, ...)."""
    return max(float(toxicity_ai), float(toxicity_human))


def to_label(score) -> int:
    """Paper score -> class: <3 LESS_TOXIC, =3 AMBIGUOUS, >3 TOXIC.

    The continuous score is cut directly, not rounded first: rounding would put 2.67 and
    3.33 into "3" with the genuinely ambiguous 3.0, which is what the earlier 5-level
    model did. The parquet stores thirds as 3.333333333, so "=3" needs a tolerance.
    """
    s = float(score)
    if abs(s - 3.0) < 1e-3:
        return AMBIGUOUS
    return LESS_TOXIC if s < 3.0 else TOXIC


def load_annotated(cache_dir: str = CACHE_DIR, normalize: bool = True):
    """Return (train_df, test_df) with a 0-indexed `label` column added."""
    frames = []
    for split in ("train", "test"):
        df = pd.read_parquet(_download(split, cache_dir))
        if normalize:
            df = normalize_target_group(df)
        df["paper_score"] = [paper_score(a, h) for a, h in zip(df["toxicity_ai"], df["toxicity_human"])]
        df["label"] = df["paper_score"].map(to_label)
        frames.append(df)
    return frames[0], frames[1]


def _selftest() -> int:
    train, test = load_annotated()
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print(f"train {train.shape}  test {test.shape}")
    check(len(train) == 8960, f"train has 8,960 rows (got {len(train)})")
    check(len(test) == 940, f"test has 940 rows (got {len(test)})")
    check(len(train) + len(test) == 9900, "total is 9,900 rows")

    groups = set(train["target_group"]) | set(test["target_group"])
    check(len(groups) == 13, f"13 target groups after normalization (got {len(groups)})")
    check(groups == set(CANONICAL_GROUPS), "groups are exactly the canonical 13")

    labels = pd.concat([train["label"], test["label"]])
    check(labels.between(0, NUM_CLASSES - 1).all(), f"labels are within 0..{NUM_CLASSES - 1}")
    counts = labels.value_counts().sort_index()
    print("  class distribution:")
    for k, v in counts.items():
        print(f"    {CLASS_NAMES[k]:<10} {v:5d}  ({v / len(labels) * 100:5.1f}%)")
    # 6,095 / 403 / 3,402 — measured on this dataset revision; a different count means
    # the binning (or the data) changed.
    check(counts.to_dict() == {0: 6095, 1: 403, 2: 3402},
          f"paper bins give 6,095 / 403 / 3,402 (got {counts.to_dict()})")

    race = pd.concat([train, test])
    race = race[race["target_group"].isin(RACE_GROUPS)]
    # 5,234 = 4,758 canonical-labelled rows + 476 rows the test split spells out in
    # free text. Counting before normalization yields 4,758 and silently loses those
    # 476 — the exact failure normalize_target_group() exists to prevent.
    check(len(race) == 5234, f"race-strict subset is 5,234 rows (got {len(race)})")

    print("\nSELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    train, test = load_annotated()
    print(f"train {train.shape}  test {test.shape}")
