from pathlib import Path

from contextguard.data import Trace, make_protocol


def fake_records():
    users = ("001", "002")
    conditions = ("a", "b", "c")
    rows = []
    for label, user in enumerate(users):
        for condition in conditions:
            for index in range(20):
                rows.append(
                    Trace(
                        path=Path(f"/{user}/{condition}{index}.mat"),
                        user=user,
                        label=label,
                        condition=condition,
                        sample_index=index,
                        source_folder="synthetic",
                    )
                )
    return tuple(rows)


def test_protocol_holds_out_future_context_for_forgotten_user():
    protocol = make_protocol(fake_records(), forget_user="001", held_condition="c")
    assert len(protocol.base_train) == 65
    assert len(protocol.forget_train) == 26
    assert len(protocol.retain_train) == 39
    assert len(protocol.forget_seen_test) == 14
    assert len(protocol.forget_held_test) == 7
    assert len(protocol.retain_test) == 21
    assert all(
        not (row.user == "001" and row.condition == "c")
        for row in protocol.base_train
    )
    assert all(row.user != "001" for row in protocol.retrain)

