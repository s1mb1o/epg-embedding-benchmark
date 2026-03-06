#!/usr/bin/env bash
# Run retrieval_inspect.py across all configured embedding models.
# Usage:  ./run_retrieval_inspect.sh [--limit N]
#
# Accepts the same env vars as run_benchmark.sh:
#   OPENAI_API_KEY, COHERE_API_KEY, JINA_API_KEY, VOYAGE_API_KEY,
#   TEST_EMB_OLLAMA=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}
PHRASES_FILE="$SCRIPT_DIR/data/epg_phrases.json"
OUTPUT="$SCRIPT_DIR/results/retrieval_inspection.md"

declare -a MODELS

# SentenceTransformers
MODELS+=("st:sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
MODELS+=("st:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MODELS+=("st:sentence-transformers/distiluse-base-multilingual-cased")
MODELS+=("st:sentence-transformers/LaBSE")
MODELS+=("st:intfloat/multilingual-e5-base")
MODELS+=("st:intfloat/multilingual-e5-large")
# Instruct variants excluded: see README § Limitations.
# MODELS+=("st:intfloat/multilingual-e5-large-instruct")
MODELS+=("st:Metric-AI/armenian-text-embeddings-1")
MODELS+=("st:BAAI/bge-m3")
MODELS+=("st:Alibaba-NLP/gte-multilingual-base:trust")
MODELS+=("st:jinaai/jina-embeddings-v3:trust")

# FlagEmbedding
MODELS+=("flag:BAAI/bge-m3")

# Ollama
if [[ "${TEST_EMB_OLLAMA:-0}" == "1" ]]; then
  MODELS+=("ollama:nomic-embed-text")
  MODELS+=("ollama:bge-m3")
fi

# OpenAI
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  MODELS+=("openai:text-embedding-3-large")
fi

# Cohere
if [[ -n "${COHERE_API_KEY:-}" ]]; then
  MODELS+=("cohere:embed-v4.0")
  MODELS+=("cohere:embed-multilingual-v3.0")
fi

# Jina
if [[ -n "${JINA_API_KEY:-}" ]]; then
  MODELS+=("jina:jina-embeddings-v3")
fi

# Voyage
if [[ -n "${VOYAGE_API_KEY:-}" ]]; then
  MODELS+=("voyage:voyage-multilingual-2")
fi

echo "Running retrieval inspection with ${#MODELS[@]} models..."
echo "Output: $OUTPUT"
echo

"$PYTHON_BIN" "$SCRIPT_DIR/retrieval_inspect.py" \
  --phrases "$PHRASES_FILE" \
  --models "${MODELS[@]}" \
  --output "$OUTPUT" \
  "$@"
