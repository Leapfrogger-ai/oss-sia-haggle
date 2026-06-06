#!/usr/bin/env bash
# Load keys from .env, then run SIA on craigslist-bargains entirely on Nebius.
#
#   ./run.sh                       # max_gen=5, run_id=1
#   MAX_GEN=3 RUN_ID=2 ./run.sh    # override
#   ./run.sh --no-web              # extra args pass through to `sia run`
set -euo pipefail
cd "$(dirname "$0")"

# Export every assignment in .env into the environment (sia reads os.getenv).
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
: "${NEBIUS_API_KEY:?NEBIUS_API_KEY is not set — add it to .env (see .env.example)}"

exec python -m sia run \
  --task_dir ./tasks/craigslist-bargains \
  --meta-agent-profile nebius-meta \
  --target-agent-profile gptoss-nebius-target \
  --max_gen "${MAX_GEN:-5}" --run_id "${RUN_ID:-1}" \
  "$@"
