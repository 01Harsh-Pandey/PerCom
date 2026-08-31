# Frozen pilot decision

## Claim under test

A WiFi user-unlearning method can report near-zero deleted-label accuracy while
retaining more cross-context identity information than an exact model retrained
without that user.

## Pilot protocol

- Dataset: NTU-Fi-HumanID, 14 users, three conditions (`a`, `b`, `c`).
- Temporal split: filename indices 0--12 for training and 13--19 for testing.
- Forgotten user: train in two conditions; reserve the third as a future context.
- Models: original, exact retraining without the user, and CIU-L/UNSIR.
- Primary endpoint: contextual probe AUC difference between unlearned and exact
  retrained models.
- Utility endpoint: retained-user classification accuracy.
- Diagnostic endpoint: deleted-label accuracy, reported but not treated as proof
  of deletion.

## Go/no-go gates

1. Original held-condition identity accuracy is at least 70%.
2. CIU-L/UNSIR lowers source-condition deleted-label accuracy to at most 10%.
3. Its contextual probe AUC exceeds exact retraining by at least 0.10 in a
   preliminary set of users/conditions.
4. The result survives all 14 users, all three held conditions, and at least
   three seeds with confidence intervals.

If gate 3 fails, do not lock or sell the proposed method. Reassess the paper
claim before spending the full experiment budget.

## Source policy

Literature screening uses primary papers from ICORE A* conferences. The CIU-L
article and original dataset/method papers are still cited as necessary primary
provenance, even when they are journals.

