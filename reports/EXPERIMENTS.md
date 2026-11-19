# Experiment record

The adapter supports all three public datasets:

| Dataset | Task | Metric |
|---|---|---|
| A | classification | Macro-F1, higher is better |
| B | regression | RMSE in minutes, lower is better |
| C | regression | RMSE in cycles, lower is better |

## Robust acceptance policy

v3 accepted small public-validation improvements that did not generalize to
hidden stages, especially for B and C. The revised policy requires:

- A: Macro-F1 at least +0.01 and accuracy not lower;
- B: RMSE at least 5% lower and MAE not higher;
- C: RMSE at least 2% lower and MAE not higher.

When the threshold is not met, the platform parent checkpoint is retained.
This guarantees that weak or noisy candidates cannot reduce the submitted
artifact below the parent baseline.

## A: gas classification

| Metric | Parent | Local candidate |
|---|---:|---:|
| Macro-F1 | 0.714030 | **0.725782** |
| Accuracy | 0.878617 | **0.885048** |

The selected A candidate used 5 epochs at learning rate `0.003`.
A follow-up 10-epoch experiment at `0.001` improved Macro-F1 to `0.726366`.

## C: turbofan RUL

| Metric | Parent | Local candidate |
|---|---:|---:|
| RMSE (cycles) | 52.868573 | **52.225263** |
| MAE (cycles) | 38.899830 | **37.826415** |

The C candidate used 15 epochs at learning rate `0.002` on 113,089 training
rows and 23,049 validation rows.

## B: bearing RUL

All B experiments use the public split:

- Training rows: 7,032
- Validation rows: 1,679
- Input features: 34
- Quantity: remaining minutes to the last observed lifecycle snapshot
- Primary metric: RMSE in minutes, lower is better

## Baseline

| Metric | Baseline |
|---|---:|
| RMSE | 681.159833 |
| MAE | 563.105841 |

## Round decisions

The policy starts from the parent checkpoint every round and keeps the parent
unless current public validation RMSE decreases.

| Round | Epochs | Learning rate | RMSE before | RMSE after | Decision |
|---:|---:|---:|---:|---:|---|
| 1 | 20 | 0.00003 | 681.159833 | 657.881581 | select candidate |
| 2 | 10 | 0.00001 | 657.881581 | 658.724751 | keep parent |
| 3 | 10 | 0.00003 | 657.881581 | 658.954165 | keep parent |
| 4 | 20 | 0.00001 | 657.881581 | 658.831068 | keep parent |
| 5 | 5 | 0.00001 | 657.881581 | 658.375229 | keep parent |
| 6 | 10 | 0.00003 | 657.881581 | 658.954165 | keep parent |
| 7 | 5 | 0.00005 | 657.881581 | 658.653667 | keep parent |
| 8 | 20 | 0.00001 | 657.881581 | 658.831068 | keep parent |
| 9 | 5 | 0.00003 | 657.881581 | 658.608592 | keep parent |
| 10 | 10 | 0.00001 | 657.881581 | 658.724751 | keep parent |

## Selected model

| Metric | Selected checkpoint |
|---|---:|
| RMSE | 657.881581 |
| MAE | 546.296365 |
| Parameters | 865 |
| Checkpoint SHA-256 | `64A1595E3750A0098044F94E68C84368936F319B06B00ED01BF89C9821D07A89` |

The selected model was trained for 20 epochs at learning rate `0.00003`.
Rounds 2–10 demonstrate the history-driven fallback to the selected parent.

## Reproduction

```sh
python -m pip install -r requirements.txt
python -m pip install ./sdk
python -m baselines.train --profile B \
  --train ./public/train.jsonl \
  --validation ./public/validation.jsonl \
  --from-baseline ./checkpoint.npz \
  --epochs 20 --learning-rate 0.00003 \
  --output ./round-1 --stage-id round-1
```

Public training and validation files are provided by the contest package and
are intentionally not committed here.
