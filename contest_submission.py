"""Harness adapter for profile B: bearing remaining-life regression in minutes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from baselines.model import Model
from baselines.predict import predict as run_prediction
from baselines.train import train_run

TASKS = ("regression",)
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
PROFILE = "B"
SEED = 42

_MODEL_CACHE: dict[str, Model] = {}

# A deliberately conservative ten-round schedule. Each round keeps the parent
# unless the current public validation RMSE improves.
EXPERIMENTS = (
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
)


def _write_jsonl(rows, destination: Path) -> int:
    count = 0
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
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
        "baselines/configs/B.json",
    )
    for relative in relative_paths:
        destination = candidate / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(app_root / relative, destination)
    return list(relative_paths)


def train(rnd) -> None:
    """Continue the platform parent checkpoint on the current public split."""
    history = rnd.history()
    history_ids = [item["id"] for item in history]
    epochs, learning_rate = EXPERIMENTS[min(rnd.round_index - 1, len(EXPERIMENTS) - 1)]
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
                    f"第 {rnd.round_index} 轮 B 题 RUL 实验提案：公开训练 "
                    f"{train_rows} 条、验证 {validation_rows} 条；从父 checkpoint 续训 "
                    f"{epochs} epoch，学习率 {learning_rate}。最近历史："
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
        PROFILE,
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
    before_rmse = report["validation_before"]["rmse"]
    after_rmse = report["validation_after"]["rmse"]
    improved = after_rmse < before_rmse
    experiment = {
        "round_index": rnd.round_index,
        "history_ids": history_ids,
        "model_call_ids": [receipt.call_id],
        "hypothesis": "保守学习率续训可在不破坏父模型的前提下降低公开验证 RMSE。",
        "epochs": epochs,
        "learning_rate": learning_rate,
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
        "profile": PROFILE,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": report["batch_size"],
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "validation_rmse_before": before_rmse,
        "validation_rmse_after": after_rmse,
        "validation_mae": report["validation_after"]["mae"],
        "decision": experiment["decision"],
    }

    submitted = rnd.submit_candidate(
        candidate,
        hypothesis=experiment["hypothesis"],
        change_summary=(
            f"第 {rnd.round_index} 轮从父 checkpoint 续训 {epochs} epoch，"
            f"学习率 {learning_rate}，并比较公开验证 RMSE。"
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
        hypothesis=experiment["hypothesis"],
        change_summary=(
            f"第 {rnd.round_index} 轮执行 {epochs} epoch、学习率 "
            f"{learning_rate} 的续训实验。"
        ),
        reason=(
            f"公开验证 RMSE 训练前={before_rmse:.6f}、训练后={after_rmse:.6f}；"
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
    """Load the selected artifact checkpoint and return the harness output payload."""
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
