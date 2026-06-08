#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-14B on the craigslist price-prediction SFT data — Lightning L40S.

Uses the stable transformers `Trainer` + PEFT path (no TRL) for version robustness, with
manual completion masking (loss only on the assistant's JSON answer). Saves a LoRA adapter
to ./haggle-adapter that vLLM serves with --enable-lora.

  python demo/lightning/finetune_lora.py            # 3 epochs, ~30 min on L40S
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parents[2]
BASE = "Qwen/Qwen3-14B"   # ungated on HF
MAXLEN = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=str(ROOT / "haggle-adapter"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tmpl(msgs, gen):
        # enable_thinking=False so Qwen3's target is the direct JSON, not <think>…</think>
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

    def encode(ex):
        msgs = ex["messages"]
        p = tok(tmpl(msgs[:-1], True), add_special_tokens=False)["input_ids"]
        f = tok(tmpl(msgs, False), add_special_tokens=False)["input_ids"][:MAXLEN]
        cut = min(len(p), len(f))
        labels = [-100] * cut + f[cut:]            # mask prompt; train on answer only
        return {"input_ids": f, "labels": labels[:len(f)], "attention_mask": [1] * len(f)}

    d = ROOT / "tasks/craigslist-bargains/finetune"
    ds = load_dataset("json", data_files={"train": str(d / "train.jsonl"), "val": str(d / "val.jsonl")})
    ds = ds.map(encode, remove_columns=ds["train"].column_names)
    print(f"train={len(ds['train'])} val={len(ds['val'])}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters()
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=str(ROOT / "haggle-sft-run"), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", logging_steps=5, save_strategy="epoch", report_to="none")
    Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
            tokenizer=tok).train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\n✅ adapter saved to {args.out}\nNext: bash demo/lightning/serve.sh")


if __name__ == "__main__":
    main()
