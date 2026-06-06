"""Smoke-test the pydantic-ai meta/feedback engine against Nebius.

Exercises the real SIA dispatch path (run_agent -> pydantic-ai -> OpenAIChatModel on
Nebius) with the file/bash tools, to confirm tool-calling works before a full run.

Run:  python smoke_meta_nebius.py     (needs NEBIUS_API_KEY exported)
"""
import asyncio
import os
import tempfile

try:  # load .env if python-dotenv is available, so keys can live in .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import sia.agent_impls  # noqa: F401 — triggers impl registration
from sia.agent_impls.base import run_agent
from sia.profiles import load_meta_agent_profile

p = load_meta_agent_profile("nebius-meta")
workdir = tempfile.mkdtemp(prefix="sia_smoke_")
print(f"impl={p.agent_impl} model={p.model} provider={p.provider.provider_id} @ {p.provider.base_url}")
print(f"workdir={workdir}")

asyncio.run(
    run_agent(
        model_name=p.model,
        max_turns="6",
        prompt="Use your write_file tool to create a file named ok.txt whose contents are exactly: PONG. Then stop.",
        agent_working_directory=workdir,
        agent_impl=p.agent_impl,
        provider=p.provider,
    )
)

target = os.path.join(workdir, "ok.txt")
if os.path.exists(target):
    print(f"\n✅ SUCCESS — tool call worked. ok.txt = {open(target).read()!r}")
else:
    print("\n❌ FAILED — agent ran but did not write ok.txt (tool-calling likely unsupported for this model)")
