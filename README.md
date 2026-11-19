# Contest Harness A/B/C

Submission for `contest.harness.workspace.v2`. The same adapter detects the
selected dataset and task:

- A: six-class gas classification
- B: bearing RUL regression in minutes
- C: turbofan RUL regression in cycles

## Models

- A: 128 inputs, 32 tanh hidden units, 6-class softmax
- B: 34 inputs, 24 tanh hidden units, 1 linear output
- C: 24 inputs, 32 tanh hidden units, 1 linear output
- All models use CPU NumPy and Adam continuation from the platform parent
  checkpoint.

Local public-validation results:

| Dataset | Metric | Baseline | Best local |
|---|---|---:|---:|
| A | Macro-F1 | 0.714030 | 0.725782 |
| B | RMSE | 681.159833 | 657.881581 |
| C | RMSE | 52.868573 | 52.225263 |

## Harness policy

`contest_submission.py` reads `rnd.history()`, selects profile A/B/C from
`rnd.dataset_key`, runs the scheduled experiment, compares the task metric
before and after training, and selects the new candidate only when it improves.
Otherwise it keeps the parent artifact.

Every round records its history IDs, model call ID, metrics, and decision in
`experiment.json`.

## Layout

```text
harness.py
contest_submission.py
baselines/
model/checkpoint.npz
model/checkpoint_A.npz
model/checkpoint_C.npz
reports/
sdk/
Dockerfile
requirements.txt
```

## Local checks

```sh
python -m pip install -r requirements.txt
python -m pip install ./sdk
python harness.py describe
python -m unittest discover -s baselines/tests -v
```

Two packaging tests in the downloaded baseline package reference
`scripts/package_baselines.py`, which is not included in the public archive.
They are unrelated to model loading or continuation. The remaining training,
checkpoint, prediction, and schema tests pass.

## Build the image

```sh
docker build --platform linux/amd64 -t contest-harness-multitask:1.0.4-20261119 .
docker save -o contest-harness-multitask-linux-amd64.tar \
  contest-harness-multitask:1.0.4-20261119
```

The image entrypoint is:

```text
python3 /app/harness.py
```

It supports:

```text
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

No PyTRiO key, hidden labels, or host credentials are included.

## Experiments

See [reports/EXPERIMENTS.md](reports/EXPERIMENTS.md) for the ten-round record.
Metrics are development results on the public validation split and are not
official hidden-test scores.
