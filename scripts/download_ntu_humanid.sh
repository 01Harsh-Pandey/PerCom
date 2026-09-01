#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data}"
mkdir -p "$destination"
cd "$destination"

if [[ ! -f NTU-Fi-HumanID.zip ]]; then
  python -m gdown "https://drive.google.com/uc?id=1IKTg5M7vDdZPnt6649i2Z3jiR4Fivsy1" -O NTU-Fi-HumanID.zip
fi

if [[ ! -d NTU-Fi-HumanID ]]; then
  unzip -q NTU-Fi-HumanID.zip
fi

find NTU-Fi-HumanID -type f -name '*.mat' | wc -l

