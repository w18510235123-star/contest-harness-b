# 10-round experiment record

All experiments use the public B split:

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
