# Contest Harness B: Bearing RUL

Profile B submission for `contest.harness.workspace.v2`. The industrial model
predicts remaining useful life in minutes from 34 horizontal/vertical bearing
features.

## Model

- CPU NumPy MLP: 34 inputs, 24 tanh hidden units, 1 linear output
- 865 parameters
- Adam continuation from the platform parent checkpoint
- Public validation RMSE: 657.881581 minutes
- Public validation MAE: 546.296365 minutes

The supplied baseline had RMSE 681.159833 and MAE 563.105841.

## Harness policy

`contest_submission.py` reads `rnd.history()`, runs the scheduled experiment,
compares public validation RMSE before and after training, and selects the new
candidate only when it improves. Otherwise it keeps the parent artifact.

Every round records its history IDs, model call ID, metrics, and decision in
`experiment.json`.

## Layout

```text
harness.py
contest_submission.py
baselines/
model/checkpoint.npz
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
docker build --platform linux/amd64 -t contest-harness-b:1.0.4-20261119 .
docker save -o contest-harness-b-linux-amd64.tar \
  contest-harness-b:1.0.4-20261119
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
