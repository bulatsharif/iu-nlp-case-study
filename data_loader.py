"""MT-Bench and ru-arena-hard loaders with local caching."""
from pathlib import Path
import json
import urllib.request

DATA_DIR = Path(__file__).parent / "data"
MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
MT_BENCH_PATH = DATA_DIR / "mt_bench.jsonl"
RU_ARENA_PATH = DATA_DIR / "ru_arena_hard.jsonl"

N_PROMPTS = 80
SEED = 42


def _ensure_mt_bench() -> None:
    if MT_BENCH_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading MT-Bench → {MT_BENCH_PATH}")
    urllib.request.urlretrieve(MT_BENCH_URL, MT_BENCH_PATH)


def _ensure_ru_arena() -> None:
    if RU_ARENA_PATH.exists():
        return
    from huggingface_hub import hf_hub_download
    import random

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading ru-arena-hard → {RU_ARENA_PATH}")
    src = hf_hub_download(
        repo_id="t-tech/ru-arena-hard",
        filename="data/question.jsonl",
        repo_type="dataset",
    )
    rows = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    random.Random(SEED).shuffle(rows)
    with RU_ARENA_PATH.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def load_mt_bench() -> list[dict]:
    """Return [{prompt, category}] for MT-Bench first turns (len == 80)."""
    _ensure_mt_bench()
    out = []
    with MT_BENCH_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            out.append({"prompt": q["turns"][0], "category": q["category"]})
    return out[:N_PROMPTS]


def load_ru_arena_hard() -> list[dict]:
    """Return [{prompt, category}] for ru-arena-hard first turns (len == 80)."""
    _ensure_ru_arena()
    out = []
    with RU_ARENA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            out.append({"prompt": x["turns"][0]["content"], "category": x["cluster"]})
    return out[:N_PROMPTS]


def load(dataset: str) -> list[dict]:
    if dataset == "mt_bench":
        return load_mt_bench()
    if dataset == "ru_arena_hard":
        return load_ru_arena_hard()
    raise ValueError(f"unknown dataset {dataset!r}")


if __name__ == "__main__":
    en = load_mt_bench()
    ru = load_ru_arena_hard()
    print(f"mt_bench: {len(en)} prompts, categories={sorted({x['category'] for x in en})}")
    print(f"ru_arena_hard: {len(ru)} prompts, categories={sorted({x['category'] for x in ru})}")
