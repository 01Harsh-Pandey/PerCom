# Frozen pilot decision

## Claim under test

A WiFi user-unlearning method can report near-zero deleted-label accuracy while
retaining more cross-context identity information than an exact model retrained
without that user.

## Pilot protocol

- Dataset: NTU-Fi-HumanID, 14 users, three conditions (`a`, `b`, `c`).
- Temporal split: filename indices 0--12 for training and 13--19 for testing.
- Every user: train in two conditions; reserve the third as a future context.
- Calibration: indices 11--12 in source conditions. Held-condition samples are
  not inspected while selecting unlearning settings.
- Models: original, exact retraining without the user, and CIU-L/UNSIR.
- Primary endpoint: predictive-distribution agreement between the unlearned and
  exact-retrained models on the forgotten user in the held condition.
- Privacy diagnostic: loss-based former-member inference compared with exact
  retraining.
- Utility endpoint: retained-user classification accuracy.
- Diagnostic endpoint: deleted-label accuracy, reported but not treated as proof
  of deletion.

## Go/no-go gates

1. The balanced original model recognizes the forgotten user and retained users
   on source-condition validation data.
2. CIU-L/UNSIR lowers source-condition deleted-label accuracy to at most 10%
   while reducing retained validation accuracy by no more than 5 points.
3. After settings are frozen, test whether the selected model differs from
   exact retraining on held-condition outputs or former-member inference.
4. Any claimed result survives all 14 users, all three held conditions, and at least
   three seeds with confidence intervals.

If gate 3 shows no meaningful difference, do not lock or sell the proposed method. Reassess the paper
claim before spending the full experiment budget.

## Source policy

Literature screening uses primary papers from ICORE A* conferences. The CIU-L
article and original dataset/method papers are still cited as necessary primary
provenance, even when they are journals.
