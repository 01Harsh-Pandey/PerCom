#!/usr/bin/env bash
#SBATCH --job-name=percom-mechanism-audit
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

ulimit -n 65536
printf 'FILE_LIMIT_SOFT=%s\n' "$(ulimit -Sn)"
printf 'FILE_LIMIT_HARD=%s\n' "$(ulimit -Hn)"

module load miniconda/default
module load cuda/12.8
source "$(conda info --base)/etc/profile.d/conda.sh"

repo_dir="${PERCOM_REPO_DIR:-$PWD}"
env_dir="${PERCOM_ENV_DIR:-$repo_dir/.venv-cu128}"
data_root="${PERCOM_DATA_ROOT:-$repo_dir/data/NTU-Fi-HumanID}"
sweep_output="${PERCOM_SWEEP_OUTPUT:?PERCOM_SWEEP_OUTPUT is required}"
output_root="${PERCOM_OUTPUT_ROOT:?PERCOM_OUTPUT_ROOT is required}"

cd "$repo_dir"

conda run --prefix "$env_dir" python -c \
  'import torch; assert torch.__version__.startswith("2.7.1+"); assert torch.version.cuda == "12.8"; print(torch.__version__, torch.version.cuda)'

conda run --no-capture-output --prefix "$env_dir" python -m contextguard.mechanism \
  --data-root "$data_root" \
  --sweep-output "$sweep_output" \
  --output "$output_root" \
  --batch-size 32 \
  --workers 8 \
  --device cuda
