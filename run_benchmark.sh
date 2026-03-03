#!/usr/bin/env bash
#/
# Integration harness for running benchmark.py across multiple
# embedding backends. Usage guidelines:
#   1. Ensure dependencies are installed in your venv:
#        pip install sentence-transformers FlagEmbedding requests
#        # Optional remote-code models (e.g. jinaai/jina-embeddings-v3)
#        # require passing --trust-remote-code; this script handles it.
#        # Optional for OpenAI results
#        export OPENAI_API_KEY=...
#        # Optional for Ollama results
#        export TEST_EMB_OLLAMA=1  # requires `ollama serve`
#   2. Execute from the repo root (script autodetects paths):
#        ./run_benchmark.sh
#   3. Override Python interpreter via PYTHON_BIN if needed:
#        PYTHON_BIN=python ./run_benchmark.sh
# Output is grouped per backend/model pair; review mean cosine and
# timing stats for cross-language quality comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}
SCRIPT_PATH="$SCRIPT_DIR/benchmark.py"
CSV_FILE="$SCRIPT_DIR/results/benchmark_results.csv"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "benchmark.py not found next to run_benchmark.sh" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/results"
rm -f "$CSV_FILE"

declare -a RUNS

# SentenceTransformers backbones (multilingual + Armenian focus)
RUNS+=("st sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
RUNS+=("st sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RUNS+=("st sentence-transformers/distiluse-base-multilingual-cased")
RUNS+=("st sentence-transformers/LaBSE")
RUNS+=("st intfloat/multilingual-e5-base")
RUNS+=("st intfloat/multilingual-e5-large")
RUNS+=("st intfloat/multilingual-e5-large-instruct")
RUNS+=("st intfloat/e5-large-v2")
RUNS+=("st intfloat/e5-large")
RUNS+=("st Metric-AI/armenian-text-embeddings-1")
RUNS+=("st BAAI/bge-m3")
RUNS+=("st all-MiniLM-L6-v2")
#RUNS+=("st jinaai/jina-embeddings-v3 --trust-remote-code")

# Known missing / experimental repos (uncomment if available upstream)
# RUNS+=("st ai-community/bge-m3")  # NOTE: repository currently not published
# RUNS+=("st intfloat/e5-large-instruct")  # NOTE: deprecated/absent on HF

# FlagEmbedding (BGE-M3) if library is installed
RUNS+=("flag BAAI/bge-m3")

# Ollama can be noisy if server missing; enable via TEST_EMB_OLLAMA=1
if [[ "${TEST_EMB_OLLAMA:-0}" == "1" ]]; then
  RUNS+=("ollama nomic-embed-text")
  RUNS+=("ollama bge-large")
  RUNS+=("ollama bge-m3")
fi

# OpenAI only when API key is present
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  RUNS+=("openai text-embedding-3-large")
fi

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "No embedding runs configured. Set OPENAI_API_KEY or TEST_EMB_OLLAMA=1 if desired." >&2
  exit 1
fi

for entry in "${RUNS[@]}"; do
  # Split entry into arguments
  read -r api model extra <<<"$entry"

  echo "================================================================"
  running_msg="Running: api=$api model=$model"
  if [[ -n "${extra:-}" ]]; then
    running_msg+=" ${extra}"
  fi
  printf '\033[32m%s\033[0m\n' "$running_msg"
  echo "----------------------------------------------------------------"

  if [[ -n "${extra:-}" ]]; then
    "$PYTHON_BIN" "$SCRIPT_PATH" --api "$api" --model "$model" --csv-file "$CSV_FILE" $extra
  else
    "$PYTHON_BIN" "$SCRIPT_PATH" --api "$api" --model "$model" --csv-file "$CSV_FILE"
  fi

  echo
done

echo "All embedding evaluations completed."
echo "Results CSV → $CSV_FILE"
