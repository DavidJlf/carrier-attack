#!/usr/bin/env bash
set -euo pipefail

forbidden='\.(png|jpe?g|webp|gif|bmp|tiff?|safetensors|ckpt|pt|pth|bin|onnx|xlsx?|zip|tar|tgz|gz)$'
bad_files="$(git ls-files | grep -Eai "$forbidden" || true)"
bad_paths="$(git ls-files | grep -E '(^|/)(data|references|subject_samples|outputs|results|runs|logs|weights|loras|checkpoints)/' || true)"
secret_hits="$(git grep -InE '(hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})' -- . ':(exclude)scripts/audit_repository.sh' || true)"

if [[ -n "$bad_files$bad_paths$secret_hits" ]]; then
  echo "Repository audit failed." >&2
  [[ -z "$bad_files" ]] || { echo "Forbidden tracked file types:" >&2; echo "$bad_files" >&2; }
  [[ -z "$bad_paths" ]] || { echo "Forbidden tracked paths:" >&2; echo "$bad_paths" >&2; }
  [[ -z "$secret_hits" ]] || { echo "Possible credentials:" >&2; echo "$secret_hits" >&2; }
  exit 1
fi

echo "Repository audit passed. Configuration CSV/JSON files are allowed; private assets and generated outputs remain excluded."
