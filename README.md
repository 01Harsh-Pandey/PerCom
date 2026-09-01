# ContextGuard: PerCom 2027 feasibility study

This repository tests one narrow question before the paper idea is locked:

> Does CIU-L-style user unlearning still leak a person's identity when that
> person later appears under a different WiFi sensing condition?

The pilot compares the unlearned model against **exact retraining without the
user**. Near-zero deleted-label accuracy is treated only as a diagnostic, not as
evidence that the identity was removed.

## First cluster run

```bash
git clone --branch codex/contextguard-pilot --single-branch \
  https://github.com/01Harsh-Pandey/PerCom.git
cd PerCom
bash scripts/download_ntu_humanid.sh data
sbatch slurm/pilot.sbatch
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f slurm-<JOB_ID>.out
```

For the corrected validation-only calibration run, use:

```bash
sbatch slurm/calibrate.sbatch
```

The calibration result is written below
`outputs/calibration-user001-held-c-seed1/`.

## Pilot v1 status

The first completed run was technically successful but scientifically invalid:

- UNSIR retained accuracy collapsed from 94.87% to 20.88%.
- The forgotten user was under-represented in the original training set.
- A separately trained representation probe achieved higher AUC on exact
  retraining than on the original model, proving it measures generic identity
  separability rather than retained training influence.

The corrected protocol holds out the same condition for every user, selects
UNSIR settings using source-condition validation data only, and evaluates the
held condition once after selection. Exact-retraining output equivalence and a
loss-based membership diagnostic replace probe-AUC difference as the primary
audits.

## Why this is not yet a paper claim

The run is a falsification pilot. We proceed only if it shows a meaningful gap
between CIU-L/UNSIR and exact retraining. A complete paper requires all 14
users, all three held conditions, multiple seeds, a second dataset, statistical
intervals, and a mitigation that closes the measured gap.

## Authentic sources

- CIU-L primary article: https://doi.org/10.1016/j.pmcj.2024.101947
- Public SenseFi code/data loader: https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- Public UNSIR implementation: https://github.com/vikram2000b/Fast-Machine-Unlearning
- WiMANS, ECCV 2024 (ICORE A*): https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5826_ECCV_2024_paper.php
- CrossSense, MobiCom 2018 (ICORE A*): https://doi.org/10.1145/3241539.3241570
- Machine Unlearning/SISA, IEEE S&P 2021 (ICORE A*): https://doi.org/10.1109/SP40001.2021.00019

The model architecture is a compact adaptation of the MIT-licensed SenseFi
NTU-Fi LeNet. The targeted-noise impair/repair implementation follows the public
UNSIR algorithm used by CIU-L.
