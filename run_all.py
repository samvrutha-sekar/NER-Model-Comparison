


import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results"
LOGS = RESULTS / "logs"
DEV = ROOT / "data" / "dev.jsonl"
TEST = ROOT / "data" / "test.jsonl"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

MODELS = [
    ("gliner", "gliner", None),
    ("qwen3_4b", "qwen3", "qwen3:4b"),
    ("qwen3_8b", "qwen3", "qwen3:8b"),
]

LABELS = ["PERSON", "ORG", "LOCATION", "EVENT"]
SOURCE_FILES = [
    "Harness/run.py",
    "Harness/postprocess.py",
    "Harness/Models/gliner_runner.py",
    "Harness/Models/llm_runner.py",
    "score.py",
    "run_all.py",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def command(args, capture=False):
    print("\n>", subprocess.list2cmdline([str(x) for x in args]))
    result = subprocess.run(
        [str(x) for x in args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if capture and result.stdout:
        print(result.stdout)
    if result.returncode:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )
    return result.stdout if capture else ""


def paths(model, split):
    return (
        RESULTS / f"predictions_{model}_{split}.jsonl",
        RESULTS / f"timings_{model}_{split}.json",
    )


def run_model(model, split, dataset):
    name, model_type, model_name = model
    pred, timing = paths(name, split)

    if not (pred.exists() and timing.exists()):
        args = [
            PYTHON, ROOT / "Harness" / "run.py",
            "--data", dataset, "--model", model_type,
        ]
        if model_name:
            args += ["--model-name", model_name]
        command(args)

    if not pred.exists() or not timing.exists():
        raise FileNotFoundError(f"Missing outputs for {name} / {split}")

    with open(timing, encoding="utf-8") as f:
        timing_data = json.load(f)

    if timing_data.get("prediction_errors", 0) != 0:
        raise RuntimeError(
            f"{name} reported prediction errors on {split}. "
            "Fix the errors and rerun before using these results."
        )

    return pred, timing, timing_data


def score_model(model_name, split, dataset, pred):
    output = command(
        [PYTHON, ROOT / "score.py", dataset, pred],
        capture=True,
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"score_{model_name}_{split}.txt").write_text(
        output, encoding="utf-8"
    )

    metrics = {}
    pattern = (
        r"^(PERSON|ORG|LOCATION|EVENT|MICRO-ALL)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)"
    )
    for match in re.finditer(pattern, output, re.MULTILINE):
        label = match.group(1)
        metrics[label] = {
            "precision": float(match.group(2)),
            "recall": float(match.group(3)),
            "f1": float(match.group(4)),
            "tp": int(match.group(5)),
            "fp": int(match.group(6)),
            "fn": int(match.group(7)),
        }

    if "MICRO-ALL" not in metrics:
        raise RuntimeError(f"Unable to parse scoring output for {model_name}")

    return metrics


def normalized_entities(items):
    result = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        label = item.get("label")
        if isinstance(text, str) and isinstance(label, str):
            result.add((text.strip().casefold(), label.strip().upper()))
    return result


def error_analysis(model_name, split, dataset, pred):
    gold_rows = read_jsonl(dataset)
    pred_rows = read_jsonl(pred)
    pred_by_id = {str(row["id"]): row for row in pred_rows}
    errors = []

    for gold in gold_rows:
        item_id = str(gold["id"])
        predicted = pred_by_id.get(item_id, {})
        gold_entities = normalized_entities(gold.get("entities", []))
        predicted_entities = normalized_entities(predicted.get("entities", []))

        for text, label in sorted(predicted_entities - gold_entities):
            errors.append({
                "dataset": split, "model": model_name, "id": item_id,
                "error": "FALSE_POSITIVE", "text": text, "label": label,
            })
        for text, label in sorted(gold_entities - predicted_entities):
            errors.append({
                "dataset": split, "model": model_name, "id": item_id,
                "error": "FALSE_NEGATIVE", "text": text, "label": label,
            })

    return errors


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(split, dataset):
    rows = []
    errors = []

    for model in MODELS:
        name = model[0]
        print(f"\n--- {split.upper()}: {name} ---")
        pred, timing_path, timing = run_model(model, split, dataset)
        metrics = score_model(name, split, dataset, pred)

        row = {
            "dataset": split,
            "model": name,
            "snippets": timing.get("snippet_count"),
            "prediction_errors": timing.get("prediction_errors", 0),
            "median_latency_seconds": timing.get("median_latency_seconds"),
            "p95_latency_seconds": timing.get("p95_latency_seconds"),
        }
        for label in LABELS + ["MICRO-ALL"]:
            if label in metrics:
                m = metrics[label]
                prefix = "micro" if label == "MICRO-ALL" else label.lower()
                row[f"{prefix}_precision"] = m["precision"]
                row[f"{prefix}_recall"] = m["recall"]
                row[f"{prefix}_f1"] = m["f1"]

        rows.append(row)
        errors.extend(error_analysis(name, split, dataset, pred))

    write_csv(RESULTS / f"{split}_comparison.csv", rows)
    (RESULTS / f"{split}_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    write_csv(RESULTS / f"{split}_error_analysis.csv", errors)
    return rows


def write_final_report(dev_rows, test_rows):
    lines = [
        "# NER Model Comparison Report",
        "",
        f"Generated (UTC): {now()}",
        "",
        "## Method",
        "",
        "Models: GLiNER, Qwen3 4B, and Qwen3 8B.",
        "All models use the project's shared postprocessing and unchanged score.py.",
        "Development data is used for development-stage comparison; test data is held out.",
        "Latency is measured by the runner and excludes warm-up calls.",
        "",
        "## Development results",
        "",
        "| Model | Micro precision | Micro recall | Micro F1 | Median latency (s) | P95 latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def table_rows(rows):
        for row in rows:
            lines.append(
                f"| {row['model']} | {row.get('micro_precision', 0):.3f} "
                f"| {row.get('micro_recall', 0):.3f} "
                f"| {row.get('micro_f1', 0):.3f} "
                f"| {row.get('median_latency_seconds', 0):.3f} "
                f"| {row.get('p95_latency_seconds', 0):.3f} |"
            )

    table_rows(dev_rows)
    lines += [
        "",
        "## Held-out test results",
        "",
        "| Model | Micro precision | Micro recall | Micro F1 | Median latency (s) | P95 latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    table_rows(test_rows)

    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- Use held-out test results for final performance claims.",
        "- The development dataset contains only a small number of snippets; results may be unstable.",
        "- Latency depends on hardware, model warm-up, and system load.",
        "- Peak memory was not measured by this script.",
        "- Monetary cost was not calculated because deployment prices and a deployment setup were not provided.",
        "- Error analysis CSV files list false positives and false negatives by model and snippet.",
        "- Review dataset quality, label definitions, and model outputs before submission.",
        "",
        "## Generated artifacts",
        "",
        "- `dev_comparison.csv` and `dev_comparison.json`",
        "- `test_comparison.csv` and `test_comparison.json`",
        "- `dev_error_analysis.csv` and `test_error_analysis.csv`",
        "- `frozen_config.json`",
        "- `test_run_manifest.json`",
        "",
        "## AI-use disclosure",
        "",
        "Complete this section accurately according to the assistance and tools used in the project.",
        "",
    ]
    (RESULTS / "FINAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    for path in (PYTHON, DEV, TEST, ROOT / "score.py"):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    manifest = RESULTS / "test_run_manifest.json"
    if manifest.exists():
        raise RuntimeError(
            f"{manifest} already exists. The test evaluation may already "
            "have started or completed. Inspect it before considering any rerun."
        )

    print("\n========== DEVELOPMENT EVALUATION ==========")
    dev_rows = build_report("dev", DEV)

    # Freeze the exact code and dataset versions before any test inference.
    frozen = {
        "created_utc": now(),
        "models": [
            {"name": name, "runner": runner, "model_name": model_name}
            for name, runner, model_name in MODELS
        ],
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in SOURCE_FILES
            if (ROOT / name).exists()
        },
        "dev_dataset_sha256": sha256(DEV),
        "test_dataset_sha256": sha256(TEST),
        "note": "Do not tune models or postprocessing based on test results.",
    }
    (RESULTS / "frozen_config.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )

    manifest.write_text(
        json.dumps({
            "started_utc": now(),
            "status": "in_progress",
            "warning": "If interrupted, inspect test outputs before any rerun.",
        }, indent=2),
        encoding="utf-8",
    )

    try:
        print("\n========== HELD-OUT TEST EVALUATION ==========")
        test_rows = build_report("test", TEST)
        write_final_report(dev_rows, test_rows)

        manifest.write_text(
            json.dumps({
                "status": "completed",
                "completed_utc": now(),
                "models": [m[0] for m in MODELS],
                "frozen_config": "Results/frozen_config.json",
            }, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        manifest.write_text(
            json.dumps({
                "status": "failed_or_interrupted",
                "updated_utc": now(),
                "error": str(exc),
                "warning": "Inspect partial test outputs before deciding on reruns.",
            }, indent=2),
            encoding="utf-8",
        )
        raise

    print("\n========== FINAL TEST SUMMARY ==========")
    for row in test_rows:
        print(
            f"{row['model']:<12} "
            f"P={row.get('micro_precision', 0):.3f} "
            f"R={row.get('micro_recall', 0):.3f} "
            f"F1={row.get('micro_f1', 0):.3f} "
            f"median={row.get('median_latency_seconds', 0):.3f}s"
        )

    print("\nWorkflow completed.")
    print("Review Results/FINAL_REPORT.md")
    print("Review Results/test_comparison.csv")
    print("Review Results/test_error_analysis.csv")
    print("Note: peak memory and monetary cost are not measured here.")


if __name__ == "__main__":
    main()
