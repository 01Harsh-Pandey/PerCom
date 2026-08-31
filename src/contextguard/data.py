from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


NTU_MEAN = 42.3199
NTU_STD = 4.9802
EXPECTED_CONDITIONS = ("a", "b", "c")


@dataclass(frozen=True)
class Trace:
    path: Path
    user: str
    label: int
    condition: str
    sample_index: int
    source_folder: str


@dataclass(frozen=True)
class Protocol:
    """One leave-one-condition-out user-deletion experiment."""

    base_train: tuple[Trace, ...]
    retrain: tuple[Trace, ...]
    forget_train: tuple[Trace, ...]
    retain_train: tuple[Trace, ...]
    forget_seen_test: tuple[Trace, ...]
    forget_held_test: tuple[Trace, ...]
    retain_test: tuple[Trace, ...]
    probe_train: tuple[Trace, ...]
    probe_test: tuple[Trace, ...]
    forget_label: int
    forget_user: str
    held_condition: str


def _parse_stem(stem: str) -> tuple[str, int]:
    if len(stem) < 2 or stem[0] not in EXPECTED_CONDITIONS:
        raise ValueError(f"Unexpected NTU-Fi filename: {stem!r}")
    try:
        index = int(stem[1:])
    except ValueError as exc:
        raise ValueError(f"Unexpected NTU-Fi filename: {stem!r}") from exc
    return stem[0], index


def scan_ntu_humanid(root: str | Path) -> tuple[Trace, ...]:
    """Scan the processed NTU-Fi-HumanID tree without trusting folder names.

    The public benchmark folders named ``test_amp`` and ``train_amp`` contain
    indices 0--12 and 13--19, respectively. We derive the temporal split from
    the filename index, which prevents an accidental train/test reversal.
    """

    root = Path(root).expanduser().resolve()
    paths = sorted(root.glob("*/*/*.mat"))
    if not paths:
        raise FileNotFoundError(
            f"No .mat files found below {root}. Expected */<user>/<a|b|c><index>.mat"
        )

    users = sorted({path.parent.name for path in paths})
    label_by_user = {user: label for label, user in enumerate(users)}
    records: list[Trace] = []
    for path in paths:
        condition, index = _parse_stem(path.stem)
        records.append(
            Trace(
                path=path,
                user=path.parent.name,
                label=label_by_user[path.parent.name],
                condition=condition,
                sample_index=index,
                source_folder=path.parent.parent.name,
            )
        )

    counts = {(user, condition): 0 for user in users for condition in EXPECTED_CONDITIONS}
    for record in records:
        counts[(record.user, record.condition)] += 1
    bad = {key: value for key, value in counts.items() if value != 20}
    if bad:
        raise ValueError(f"Unexpected per-user/per-condition counts: {bad}")
    return tuple(records)


def make_protocol(
    records: Sequence[Trace], forget_user: str, held_condition: str
) -> Protocol:
    """Build a deletion protocol with a future, unseen condition.

    For retained users, the base model sees every condition. For the user who
    will later request deletion, the base model sees only the two source
    conditions. The third condition is reserved as a future-context test.
    """

    if held_condition not in EXPECTED_CONDITIONS:
        raise ValueError(f"held_condition must be one of {EXPECTED_CONDITIONS}")
    users = sorted({record.user for record in records})
    if forget_user not in users:
        raise ValueError(f"Unknown forget user {forget_user!r}; available: {users}")
    forget_label = next(record.label for record in records if record.user == forget_user)

    early = tuple(record for record in records if record.sample_index <= 12)
    late = tuple(record for record in records if record.sample_index >= 13)

    base_train = tuple(
        record
        for record in early
        if not (record.user == forget_user and record.condition == held_condition)
    )
    forget_train = tuple(record for record in base_train if record.user == forget_user)
    retain_train = tuple(record for record in base_train if record.user != forget_user)
    retrain = retain_train
    forget_seen_test = tuple(
        record
        for record in late
        if record.user == forget_user and record.condition != held_condition
    )
    forget_held_test = tuple(
        record
        for record in late
        if record.user == forget_user and record.condition == held_condition
    )
    retain_test = tuple(record for record in late if record.user != forget_user)

    # An attacker receives source-condition reference traces and attempts to
    # link a later held-condition trace. The same probe protocol is evaluated
    # on the unlearned and exact-retrained representations; their difference
    # is the contextual unlearning gap.
    probe_train = tuple(
        record
        for record in early
        if record.condition != held_condition
    )
    probe_test = tuple(
        record
        for record in late
        if record.condition == held_condition
    )

    if not forget_train or not forget_held_test:
        raise ValueError("Protocol produced an empty forget split")
    return Protocol(
        base_train=base_train,
        retrain=retrain,
        forget_train=forget_train,
        retain_train=retain_train,
        forget_seen_test=forget_seen_test,
        forget_held_test=forget_held_test,
        retain_test=retain_test,
        probe_train=probe_train,
        probe_test=probe_test,
        forget_label=forget_label,
        forget_user=forget_user,
        held_condition=held_condition,
    )


class NTUHumanIDDataset(Dataset):
    def __init__(self, records: Iterable[Trace]):
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        array = sio.loadmat(record.path)["CSIamp"].astype(np.float32, copy=False)
        if array.shape != (342, 2000):
            raise ValueError(f"Unexpected CSI shape {array.shape} in {record.path}")
        array = (array[:, ::4] - NTU_MEAN) / NTU_STD
        array = np.ascontiguousarray(array.reshape(3, 114, 500))
        return {
            "x": torch.from_numpy(array),
            "y": record.label,
            "condition": record.condition,
            "sample_index": record.sample_index,
            "user": record.user,
            "path": str(record.path),
        }


def stratified_train_validation(
    records: Sequence[Trace], validation_indices: frozenset[int] = frozenset({11, 12})
) -> tuple[tuple[Trace, ...], tuple[Trace, ...]]:
    train = tuple(record for record in records if record.sample_index not in validation_indices)
    validation = tuple(record for record in records if record.sample_index in validation_indices)
    if not train or not validation:
        raise ValueError("Empty train or validation split")
    return train, validation

