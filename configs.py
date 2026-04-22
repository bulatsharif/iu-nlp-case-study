"""Six core experiment configs for the speculative-decoding benchmark."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpConfig:
    name: str
    target_model: str
    dtype: str                # "bfloat16" or "float16"
    dataset: str              # "mt_bench" or "ru_arena_hard"
    method: str               # "baseline" | "ngram" | "eagle3"
    draft_model: Optional[str] = None
    num_speculative_tokens: int = 5
    prompt_lookup_min: int = 2
    prompt_lookup_max: int = 4
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    extra: dict = field(default_factory=dict)


QWEN = "Qwen/Qwen3-8B"
AVIBE = "AvitoTech/avibe"
QWEN_EAGLE = "AngelSlim/Qwen3-8B_eagle3"
AVIBE_EAGLE = "AvitoTech/avibe-eagle"


CONFIGS: dict[str, ExpConfig] = {
    "qwen_baseline": ExpConfig(
        name="qwen_baseline",
        target_model=QWEN, dtype="bfloat16",
        dataset="mt_bench", method="baseline",
    ),
    "qwen_ngram": ExpConfig(
        name="qwen_ngram",
        target_model=QWEN, dtype="bfloat16",
        dataset="mt_bench", method="ngram",
    ),
    "qwen_eagle3": ExpConfig(
        name="qwen_eagle3",
        target_model=QWEN, dtype="bfloat16",
        dataset="mt_bench", method="eagle3",
        draft_model=QWEN_EAGLE,
    ),
    "avibe_baseline": ExpConfig(
        name="avibe_baseline",
        target_model=AVIBE, dtype="float16",
        dataset="ru_arena_hard", method="baseline",
    ),
    "avibe_ngram": ExpConfig(
        name="avibe_ngram",
        target_model=AVIBE, dtype="float16",
        dataset="ru_arena_hard", method="ngram",
    ),
    "avibe_eagle3": ExpConfig(
        name="avibe_eagle3",
        target_model=AVIBE, dtype="float16",
        dataset="ru_arena_hard", method="eagle3",
        draft_model=AVIBE_EAGLE,
        # avibe + avibe-eagle together report ~28 GiB during vLLM's
        # model-loading phase, leaving almost no headroom for KV cache on a
        # 31.36 GiB RTX 5090. Disable cudagraphs (avoids cudagraph-memory
        # profiling OOM) and shrink the serving context so a full max_tokens
        # generation still fits in the KV cache budget. Prompts + 512 new
        # tokens stay well under 2048 on MT-Bench / ru-arena-hard.
        gpu_memory_utilization=0.97,
        enforce_eager=True,
        max_model_len=2048,
    ),
}


def get(name: str) -> ExpConfig:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    return CONFIGS[name]
