#!/usr/bin/env bash
#SBATCH --job-name=percom-pilot
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

module load miniconda/default
module load cuda/12.8

source "$(conda info --base)/etc/profile.d/conda.sh"

repo_dir="${PERCOM_REPO_DIR:-$PWD}"
env_dir="${PERCOM_ENV_DIR:-$repo_dir/.venv}"
data_root="${PERCOM_DATA_ROOT:-$repo_dir/data/NTU-Fi-HumanID}"
output_root="${PERCOM_OUTPUT_ROOT:-$repo_dir/outputs/pilot-user001-held-c-seed1}"

cd "$repo_dir"

if [[ ! -d "$env_dir/conda-meta" ]]; then
  conda create --yes --prefix "$env_dir" python=3.11
  conda run --prefix "$env_dir" python -m pip install --upgrade pip
  conda run --prefix "$env_dir" python -m pip install -r requirements.txt
  conda run --prefix "$env_dir" python -m pip install -e .
fi

conda run --no-capture-output --prefix "$env_dir" python -m contextguard.pilot \
  --data-root "$data_root" \
  --output "$output_root" \
  --forget-user 001 \
  --held-condition c \
  --seed 1 \
  --epochs 30 \
  --batch-size 32 \
  --workers 8 \
  --device cuda
