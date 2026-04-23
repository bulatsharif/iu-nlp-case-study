"""Train an EAGLE3 draft module on Llama-3.2-1B with 20k Daring-Anteater samples, 1 epoch."""
import json
import random
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import transformers

ROOT = Path(__file__).parent
# Pinned: main/ has been refactored past the {LazySupervisedDataset,
# DataCollatorWithPadding} API this script relies on (see PR #668, Feb 2026).
EAGLE_UTILS_URL = (
    "https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/"
    "0.27.0/examples/speculative_decoding/eagle_utils.py"
)
EAGLE_UTILS_PATH = ROOT / "eagle_utils.py"

BASE_MODEL = "unsloth/Llama-3.2-1B"
DATASET_DIR = Path("/tmp/Daring-Anteater")
OUTPUT_DIR = ROOT / "eagle_out"
EXPORT_DIR = ROOT / "eagle_hf_ckpt"
LOSS_JSON = ROOT / "results" / "train_loss.json"
LOSS_PLOT = ROOT / "plots" / "train_loss.png"

N_SAMPLES = 80_000
N_EPOCHS = 2
SEED = 42


def ensure_eagle_utils() -> None:
    if EAGLE_UTILS_PATH.exists():
        return
    print(f"fetching eagle_utils.py from {EAGLE_UTILS_URL}")
    urllib.request.urlretrieve(EAGLE_UTILS_URL, EAGLE_UTILS_PATH)


def ensure_dataset() -> None:
    if (DATASET_DIR / "train.jsonl").exists():
        return
    print(f"cloning Daring-Anteater → {DATASET_DIR}")
    subprocess.run(
        ["git", "clone",
         "https://huggingface.co/datasets/nvidia/Daring-Anteater",
         str(DATASET_DIR)],
        check=True,
    )


def load_and_sample() -> list[dict]:
    with (DATASET_DIR / "train.jsonl").open() as f:
        data = [json.loads(line) for line in f]
    # Real Daring-Anteater ships ~99k rows. If we somehow got orders of
    # magnitude fewer (e.g. a stale dry-run stub at DATASET_DIR shadowed
    # the real clone), abort so the user notices instead of training on
    # garbage.
    if len(data) < N_SAMPLES // 2:
        raise RuntimeError(
            f"{DATASET_DIR/'train.jsonl'} has only {len(data)} rows, expected "
            f">= {N_SAMPLES // 2}. Remove the directory and re-run so the real "
            "dataset is fetched."
        )
    random.Random(SEED).shuffle(data)
    sample = data[:N_SAMPLES]
    print(f"sampled {len(sample)} / {len(data)} rows from Daring-Anteater (seed={SEED})")
    return sample


def build_model_and_tokenizer():
    import modelopt.torch.opt as mto
    import modelopt.torch.speculative as mtsp
    from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG

    mto.enable_huggingface_checkpointing()
    # attn_implementation="eager": ModelOpt's eagle module feeds a non-standard
    # attention mask (block-diagonal for the draft layer) that torch SDPA
    # rejects across all backends on Blackwell (flash/mem-efficient/cuDNN all
    # decline, and the math fallback then errors with "No available kernel").
    # Eager is ~5-10% slower per step but is the only path that actually runs.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="cuda",
        attn_implementation="eager",
    )

    config = EAGLE3_DEFAULT_CFG["config"]
    config["eagle_architecture_config"].update({
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "draft_vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        # Eagle module ships its own LlamaDecoderLayer that ignores the base
        # model's attn_implementation; explicitly pin it to eager too,
        # otherwise it stays on sdpa and crashes (see comment above).
        "_attn_implementation": "eager",
    })
    mtsp.convert(model, [("eagle", config)])

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL, model_max_length=1024)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{%- for message in messages %}"
            "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
            "{%- endfor %}"
        )
    return model, tokenizer


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)


def extract_and_plot_loss(trainer) -> None:
    import matplotlib.pyplot as plt
    history = [e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    steps = [e.get("step", i) for i, e in enumerate(history)]
    losses = [e["loss"] for e in history]

    LOSS_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOSS_JSON.write_text(json.dumps(
        [{"step": s, "loss": l} for s, l in zip(steps, losses)], indent=2
    ))
    print(f"wrote {LOSS_JSON} ({len(losses)} entries)")

    if not losses:
        print("no training-loss entries; skipping plot")
        return
    LOSS_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, color="#1f77b4", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title(f"EAGLE3 draft on {BASE_MODEL} — {N_SAMPLES} samples, {N_EPOCHS} epoch")
    fig.tight_layout()
    fig.savefig(LOSS_PLOT, dpi=160)
    print(f"saved {LOSS_PLOT}")


def main() -> None:
    ensure_eagle_utils()
    sys.path.insert(0, str(ROOT))
    from eagle_utils import DataCollatorWithPadding, LazySupervisedDataset

    ensure_dataset()
    data = load_and_sample()
    split = int(len(data) * 0.95)

    model, tokenizer = build_model_and_tokenizer()
    train_ds = LazySupervisedDataset(data[:split], tokenizer=tokenizer)
    eval_ds = LazySupervisedDataset(data[split:], tokenizer=tokenizer)

    # Fixed-length padding: modelopt's eagle module caches the TTT attention
    # mask keyed by ttt_step ONLY (see `_cached_attn_blk_masks` in
    # modelopt.torch.speculative.plugins.transformers), so the seq_length from
    # the first batch gets baked in. Variable-length batches then hit a
    # [q_len, k_len] mismatch against the stale cached mask on ttt_step>=1.
    # Upstream worked around this with `padding="max_length"` in
    # LanguageDataCollator; we do the same by subclassing the old collator.
    class FixedLengthCollator(DataCollatorWithPadding):
        def __init__(self, pad_to: int):
            self.pad_to = pad_to

        def __call__(self, features):
            import torch as _t
            for item in features:
                for k in ("input_ids", "attention_mask", "loss_mask", "labels"):
                    if item[k].shape[0] > self.pad_to:
                        item[k] = item[k][: self.pad_to]
                # If eagle_utils' offset-mapping heuristic failed to tag any
                # assistant tokens (or long prompts truncated the assistant
                # span), loss_mask is all zeros and modelopt's eagle_loss
                # evaluates to a scalar 0 tensor. That tensor then trips
                # `eagle_loss or 0` in modelopt, collapsing `loss` to Python
                # int 0 which breaks `.backward()`. Fall back to attention_mask
                # so we at least train on all non-pad positions.
                if int(item["loss_mask"].sum()) == 0:
                    item["loss_mask"] = item["attention_mask"].clone().to(
                        item["loss_mask"].dtype
                    )
            out = {}
            for k in ("input_ids", "attention_mask", "loss_mask", "labels"):
                out[k] = _t.stack([self.paddingtensor(f[k], self.pad_to) for f in features])
            return out

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    trainer = transformers.Trainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=FixedLengthCollator(pad_to=tokenizer.model_max_length),
    )
    trainer._move_model_to_device(model, trainer.args.device)
    assert trainer.label_smoother is None, "label_smoother is not supported in speculative decoding!"

    trainer.train()
    trainer.save_state()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    extract_and_plot_loss(trainer)

    # export_speculative_decoding writes a draft-only checkpoint in the
    # deployment format vLLM expects (architecture `LlamaForCausalLMEagle3`,
    # single `layers.0.*` decoder, no `eagle_module.` prefix, no base-model
    # weights). `export_hf_checkpoint` by contrast produces a "unified" dump
    # that vLLM cannot load for EAGLE3.
    from modelopt.torch.export import export_speculative_decoding
    model.eval()
    export_speculative_decoding(model, export_dir=str(EXPORT_DIR))
    print(f"exported EAGLE3 draft checkpoint → {EXPORT_DIR}")


if __name__ == "__main__":
    main()
