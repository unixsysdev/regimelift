#!/bin/bash
# =============================================================================
# He-LMAS Bridge Upload to HuggingFace Hub
# 
# Creates a HuggingFace repo and uploads the trained Bridge checkpoint
# with full metadata and model card.
#
# Usage:
#   ./scripts/push_to_hf.sh                           # Upload bridge_final.pt
#   ./scripts/push_to_hf.sh checkpoints/bridge_step_5000.pt  # Upload specific checkpoint
# =============================================================================

set -e

# Configuration
HF_USERNAME="datasysdev"
REPO_NAME="he-lmas-bridge"
REPO_ID="${HF_USERNAME}/${REPO_NAME}"
CHECKPOINT="${1:-checkpoints/bridge_final.pt}"
BRANCH="${2:-main}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}He-LMAS Bridge → HuggingFace Hub${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Username:   ${YELLOW}${HF_USERNAME}${NC}"
echo -e "Repository: ${YELLOW}${REPO_ID}${NC}"
echo -e "Checkpoint: ${YELLOW}${CHECKPOINT}${NC}"
echo ""

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${RED}Error: huggingface_hub not installed${NC}"
    echo "Run: pip install huggingface_hub"
    exit 1
fi

# Check if logged in to HF
if ! huggingface-cli whoami 2>/dev/null | grep -q "${HF_USERNAME}"; then
    echo -e "${YELLOW}Not logged in to HuggingFace. Logging in...${NC}"
    huggingface-cli login
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: ${CHECKPOINT}${NC}"
    echo "Available checkpoints:"
    ls -la checkpoints/*.pt 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Create temporary directory for upload
UPLOAD_DIR=$(mktemp -d)
trap "rm -rf $UPLOAD_DIR" EXIT

echo -e "${GREEN}Preparing upload package...${NC}"

# Copy checkpoint
cp "$CHECKPOINT" "$UPLOAD_DIR/bridge.pt"

# Get checkpoint info
CHECKPOINT_INFO=$(python -c "
import torch
import json

ckpt = torch.load('${CHECKPOINT}', map_location='cpu')
info = {
    'step': ckpt.get('state', {}).get('step', 'unknown'),
    'samples_seen': ckpt.get('state', {}).get('samples_seen', 'unknown'),
    'config': ckpt.get('config', {})
}

# Count parameters
num_params = sum(p.numel() for p in ckpt['bridge_state_dict'].values())
info['num_parameters'] = num_params

print(json.dumps(info))
")

STEP=$(echo "$CHECKPOINT_INFO" | python -c "import sys,json; print(json.load(sys.stdin)['step'])")
NUM_PARAMS=$(echo "$CHECKPOINT_INFO" | python -c "import sys,json; print(json.load(sys.stdin)['num_parameters'])")

echo -e "  Training Step: ${YELLOW}${STEP}${NC}"
echo -e "  Parameters:    ${YELLOW}${NUM_PARAMS}${NC}"

# Create model card
cat > "$UPLOAD_DIR/README.md" << EOF
---
license: apache-2.0
tags:
  - pytorch
  - he-lmas
  - kv-cache-projection
  - reasoning-transfer
  - qwen
datasets:
  - openmath
language:
  - en
base_model:
  - Qwen/Qwen3-8B
  - Qwen/Qwen3-1.7B
pipeline_tag: text-generation
---

# He-LMAS Bridge: Qwen3-8B → Qwen3-1.7B

This is a trained **KV Cache Projection Bridge** for the He-LMAS (Heterogeneous Latent Manifold Alignment System) framework.

## What is He-LMAS?

He-LMAS enables **reasoning transfer** between large and small language models by projecting the Teacher's KV cache into the Student's geometry. Instead of expensive fine-tuning, we train only a small Bridge (~1.8M parameters) to align the attention manifolds.

## Model Details

| Property | Value |
|----------|-------|
| Teacher Model | Qwen3-8B (36 layers, 8 KV heads) |
| Student Model | Qwen3-1.7B (28 layers, 8 KV heads) |
| Bridge Parameters | ${NUM_PARAMS::-3}K |
| Training Step | ${STEP} |
| Loss Function | Attention Consistency (RFC Section 3.2) |

## Usage

\`\`\`python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")

# Load Bridge
from huggingface_hub import hf_hub_download
bridge_path = hf_hub_download(repo_id="${REPO_ID}", filename="bridge.pt")
bridge_ckpt = torch.load(bridge_path)

# Initialize Bridge (see He-LMAS repo for full implementation)
# from helmas.bridge import HeLMAS_Bridge
# bridge = HeLMAS_Bridge(...)
# bridge.load_state_dict(bridge_ckpt['bridge_state_dict'])
\`\`\`

## Training Configuration

\`\`\`json
$(echo "$CHECKPOINT_INFO" | python -c "import sys,json; print(json.dumps(json.load(sys.stdin)['config'], indent=2))")
\`\`\`

## Citation

\`\`\`bibtex
@misc{helmas2026,
  title={He-LMAS: Heterogeneous Latent Manifold Alignment System},
  author={datasysdev},
  year={2026},
  url={https://github.com/unixsysdev/regimelift}
}
\`\`\`

## License

Apache 2.0
EOF

# Create config.json
cat > "$UPLOAD_DIR/config.json" << EOF
{
  "framework": "he-lmas",
  "version": "1.0.0",
  "teacher_model": "Qwen/Qwen3-8B",
  "student_model": "Qwen/Qwen3-1.7B",
  "teacher_layers": 36,
  "student_layers": 28,
  "num_kv_heads": 8,
  "head_dim": 128,
  "num_parameters": ${NUM_PARAMS},
  "training_step": ${STEP},
  "loss_type": "attention_consistency",
  "checkpoint_file": "bridge.pt"
}
EOF

echo -e "${GREEN}Creating/updating HuggingFace repository...${NC}"

# Create repo if it doesn't exist, then upload
python << EOF
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
repo_id = "${REPO_ID}"

# Create repo if it doesn't exist
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Repo exists or created: {e}")

# Upload files
upload_dir = "${UPLOAD_DIR}"
files = os.listdir(upload_dir)

for filename in files:
    filepath = os.path.join(upload_dir, filename)
    print(f"Uploading: {filename}")
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=filename,
        repo_id=repo_id,
        commit_message=f"Upload {filename} (step ${STEP})"
    )

print("✅ Upload complete!")
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Successfully uploaded to HuggingFace!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "View your model: ${YELLOW}https://huggingface.co/${REPO_ID}${NC}"
echo ""
