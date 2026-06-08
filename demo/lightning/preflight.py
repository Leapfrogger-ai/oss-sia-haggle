#!/usr/bin/env python3
"""Fast env check — run after setup.sh, BEFORE the 30-min finetune.
Catches ABI breaks (numpy/pandas), missing libs, Trainer API drift, and the Qwen3
chat-template kwarg in ~10s instead of failing deep into training. Reports ALL
problems at once, then exits non-zero if any failed.
"""
import importlib
import inspect
import sys

ok = True
def check(label, fn):
    global ok
    try:
        print(f"  {label}: {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"  {label}: ❌ {type(e).__name__}: {e}"); ok = False

def ver(mod):
    return lambda: importlib.import_module(mod).__version__

check("numpy", ver("numpy"))
check("pandas import (ABI check)", ver("pandas"))      # pandas->numpy ABI break shows here
check("datasets import", ver("datasets"))
check("torch (+cuda)", lambda: f"{importlib.import_module('torch').__version__} "
                               f"cuda={importlib.import_module('torch').cuda.is_available()}")
check("transformers", ver("transformers"))
check("peft", ver("peft"))
check("bitsandbytes", ver("bitsandbytes"))

def trainer_kw():
    from transformers import Trainer
    return "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
check("Trainer tokenizer kwarg", trainer_kw)

def template_check():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        ids = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True,
                                      add_generation_prompt=True)
    return f"OK ({len(ids)} toks)"
check("Qwen3 tokenizer + chat template", template_check)

print("\n✅ PREFLIGHT PASSED — safe to run finetune_lora.py" if ok
      else "\n❌ PREFLIGHT FAILED — fix the ❌ lines above before finetuning")
sys.exit(0 if ok else 1)
