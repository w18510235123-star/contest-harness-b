"""Harness adapter for A classification and B/C regression."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from baselines.model import Model
from baselines.predict import predict as run_prediction
from baselines.train import train_run

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
SEED = 42

PROFILE_TASKS = {
    "A": "classification",
    "B": "regression",
    "C": "regression",
}

# Conservative ten-round schedules. A candidate is selected only when the
# current public validation metric improves over the parent checkpoint.
EXPERIMENTS = {
    "A": (
        (5, 0.003),
        (10, 0.001),
        (10, 0.003),
        (20, 0.001),
        (5, 0.0003),
        (10, 0.001),
        (5, 0.003),
        (20, 0.0003),
        (5, 0.001),
        (10, 0.0003),
    ),
    "B": (
        (20, 0.00003),
        (10, 0.00001),
        (10, 0.00003),
        (20, 0.00001),
        (5, 0.00001),
        (10, 0.00003),
        (5, 0.00005),
        (20, 0.00001),
        (5, 0.00003),
        (10, 0.00001),
    ),
    "C": (
        (15, 0.002),
        (10, 0.001),
        (10, 0.002),
        (20, 0.001),
        (5, 0.0005),
        (10, 0.001),
        (5, 0.002),
        (20, 0.0005),
        (5, 0.001),
        (10, 0.0005),
    ),
}

_MODEL_CACHE: dict[str, Model] = {}


def _write_jsonl(rows, destination: Path) -> int:
    count = 0
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def _copy_submission_code(candidate: Path) -> list[str]:
    app_root = Path(__file__).resolve().parent
    relative_paths = (
        "baselines/__init__.py",
        "baselines/data.py",
        "baselines/model.py",
        "baselines/predict.py",
        "baselines/train.py",
        "baselines/configs/A.json",
        "baselines/configs/B.json",
        "baselines/configs/C.json",
    )
    for relative in relative_paths:
        destination = candidate / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(app_root / relative, destination)
    return list(relative_paths)


def _profile_for_round(rnd) -> tuple[str, str]:
    dataset_key = str(getattr(rnd, "dataset_key", "")).upper()
    if dataset_key not in PROFILE_TASKS:
        raise ValueError(f"unsupported dataset key: {dataset_key!r}")
    expected_task = PROFILE_TASKS[dataset_key]
    task_contract = getattr(rnd, "task", None)
    task = task_contract.get("task") if isinstance(task_contract, dict) else None
    if task is not None and task != expected_task:
        raise ValueError(
            f"dataset {dataset_key} requires task {expected_task!r}, got {task!r}"
        )
    return dataset_key, expected_task


def _metric_result(
    task: str, before: dict, after: dict
) -> tuple[str, str, float, float, bool]:
    if task == "classification":
        name, direction = "macro_f1", "higher"
        before_value = float(before["macro_f1"])
        after_value = float(after["macro_f1"])
        improved = after_value > before_value
    else:
        name, direction = "rmse", "lower"
        before_value = float(before["rmse"])
        after_value = float(after["rmse"])
        improved = after_value < before_value
    return name, direction, before_value, after_value, improved


def train(rnd) -> None:
    """Continue the platform parent checkpoint on the selected dataset."""
    profile, task = _profile_for_round(rnd)
    history = rnd.history()
    history_ids = [item["id"] for item in history]
    schedule = EXPERIMENTS[profile]
    epochs, learning_rate = schedule[min(rnd.round_index - 1, len(schedule) - 1)]

    work = Path(os.environ.get("CONTEST_WORK", "/workspace")) / f"round-{rnd.round_index}"
    work.mkdir(parents=True, exist_ok=False)
    training_path = work / "train.jsonl"
    validation_path = work / "validation.jsonl"
    train_rows = _write_jsonl(rnd.data.training_rows(), training_path)
    validation_rows = _write_jsonl(rnd.data.validation_rows(), validation_path)

    parent = rnd.parent.path("checkpoint.npz")
    receipt = rnd.model(
        [
            {
                "role": "user",
                "content": (
                    f"第 {rnd.round_index} 轮 {profile} 数据集实验提案：任务 {task}，"
                    f"公开训练 {train_rows} 条、验证 {validation_rows} 条；"
                    f"从父 checkpoint 续训 {epochs} epoch，学习率 {learning_rate}。"
                    f"最近历史："
                    f"{json.dumps(history[-3:], ensure_ascii=False, default=str)[:3000]}。"
                    "请用一个词确认收到该提案。"
                ),
            }
        ],
        max_tokens=min(32, rnd.max_output_tokens),
        temperature=0,
    )

    candidate = work / "candidate"
    report = train_run(
        profile,
        training_path,
        validation_path,
        candidate,
        resume=parent,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=SEED,
        stage_id=f"round-{rnd.round_index}",
    )
    code_paths = _copy_submission_code(candidate)
    metric, direction, before_value, after_value, improved = _metric_result(
        task,
        report["validation_before"],
        report["validation_after"],
    )
    hypothesis = (
        "保守学习率续训可在不破坏父模型的前提下提高公开验证 Macro-F1。"
        if task == "classification"
        else "保守学习率续训可在不破坏父模型的前提下降低公开验证 RMSE。"
    )
    experiment = {
        "round_index": rnd.round_index,
        "dataset_key": profile,
        "task": task,
        "history_ids": history_ids,
        "model_call_ids": [receipt.call_id],
        "hypothesis": hypothesis,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "metric": metric,
        "metric_direction": direction,
        "metric_before": before_value,
        "metric_after": after_value,
        "validation_before": report["validation_before"],
        "validation_after": report["validation_after"],
        "improved": improved,
        "decision": "select_candidate" if improved else "keep_parent",
    }
    (candidate / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    code_paths.append("experiment.json")
    configuration = {
        "dataset_key": profile,
        "task": task,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": report["batch_size"],
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "metric": metric,
        "metric_direction": direction,
        "metric_before": before_value,
        "metric_after": after_value,
        "decision": experiment["decision"],
    }

    submitted = rnd.submit_candidate(
        candidate,
        hypothesis=hypothesis,
        change_summary=(
            f"第 {rnd.round_index} 轮从父 checkpoint 续训 {epochs} epoch，"
            f"学习率 {learning_rate}，并比较公开验证 {metric}。"
        ),
        configuration=configuration,
        seed=SEED,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=code_paths,
        predict_argv=PREDICT_ARGV,
    )
    selected_artifact_id = submitted["artifact_id"] if improved else rnd.parent.artifact_id
    rnd.finish(
        selected_artifact_id=selected_artifact_id,
        hypothesis=hypothesis,
        change_summary=(
            f"第 {rnd.round_index} 轮执行 {epochs} epoch、学习率 "
            f"{learning_rate} 的 {profile} 数据集续训实验。"
        ),
        reason=(
            f"公开验证 {metric} 训练前={before_value:.6f}、"
            f"训练后={after_value:.6f}；"
            + (
                "结果改善，选择本轮候选 checkpoint。"
                if improved
                else "结果未改善，保留父 checkpoint，候选仅作为历史实验记录。"
            )
        ),
        configuration=configuration,
        seed=SEED,
        history_ids=history_ids,
        model_call_ids=[receipt.call_id],
    )


def predict(sample, model_dir):
    """Load the selected artifact checkpoint and return the Harness payload."""
    checkpoint = Path(
        os.environ.get("CONTEST_ARTIFACT_PATH", "")
        or Path(model_dir) / "checkpoint.npz"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"artifact checkpoint not found: {checkpoint}")
    cache_key = str(checkpoint.resolve())
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = Model.load(checkpoint)
        _MODEL_CACHE[cache_key] = model
    if sample.task != model.meta["task"]:
        raise ValueError(
            f"prediction task {sample.task!r} does not match checkpoint task "
            f"{model.meta['task']!r}"
        )
    return run_prediction(model, sample.inputs)["outputs"]
