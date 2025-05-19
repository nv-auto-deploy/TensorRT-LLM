#!/usr/bin/env bash

# Re-run post-create command just in case...
mkdir -p $HOME/.local/bin && source $HOME/.profile && pip install -r requirements-dev.txt && pre-commit install --install-hooks

# Install additional hooks for pre-commit
curl -sSL https://gitlab-master.nvidia.com/-/snippets/8859/raw/main/install.sh | tr -d '\r' | bash

# Check if TRTLLM_PRECOMPILED_LOCATION is already defined
if [ -z "$TRTLLM_PRECOMPILED_LOCATION" ]; then
  # 1. Define the base URL and the artifacts filename
  BASE_URL="https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge"
  ARTIFACTS_FILENAME="TensorRT-LLM.tar.gz"

  # 2. Pick file URL with the highest base number that contains the tar.gz file
  FULL_URL=$( \
    curl -s "$BASE_URL/" \
    | grep -o 'href="[0-9]*/"' \
    | sed -E 's/href="([0-9]*)\/"/\1/' \
    | sort -n \
    | while read NUMBER; do \
        FILE_URL="${BASE_URL}/${NUMBER}/${ARTIFACTS_FILENAME}"; \
        if curl --head --silent --fail "$FILE_URL" > /dev/null; then \
          echo "$FILE_URL"; \
        fi; \
      done | tail -n 1 \
  )

  # 3. Check result and export if found
  if [ -n "$FULL_URL" ]; then
    export TRTLLM_PRECOMPILED_LOCATION="$FULL_URL"
    echo "Picked wheel: $TRTLLM_PRECOMPILED_LOCATION"
  else
    echo "[ERROR] No valid files found at $BASE_URL" >&2
    exit 1
  fi
else
  echo "TRTLLM_PRECOMPILED_LOCATION is already set to: $TRTLLM_PRECOMPILED_LOCATION"
fi

# 4. Run editable pip install of tensorrt_llm
pip install -e ".[devel]"
