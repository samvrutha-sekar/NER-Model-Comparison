

import argparse
import json
import statistics
import time
from pathlib import Path

from postprocess import postprocess
from Models.gliner_runner import GLiNERRunner
from Models.llm_runner import LLMRunner


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Results"


def read_jsonl(path):
    """Read a JSONL dataset."""
    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def write_jsonl(path, rows):
    """Write predictions to a JSONL file."""
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def calculate_p95(latencies):
    """Calculate the nearest-rank 95th percentile latency."""
    if not latencies:
        return None

    sorted_latencies = sorted(latencies)
    index = max(0, int(0.95 * len(sorted_latencies)) - 1)
    return sorted_latencies[index]


def main():
    parser = argparse.ArgumentParser(
        description="Run NER predictions and measure inference latency."
    )

    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "dev.jsonl"),
        help="Path to a JSONL dataset.",
    )

    parser.add_argument(
        "--model",
        choices=["gliner", "qwen3"],
        required=True,
        help="NER model to evaluate.",
    )

    parser.add_argument(
        "--model-name",
        default="qwen3:4b",
        help="Ollama model name when --model qwen3 is selected.",
    )

    args = parser.parse_args()

    data_path = Path(args.data).resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    data = read_jsonl(data_path)

    if not data:
        raise ValueError(
            f"The dataset is empty: {data_path}"
        )

    # Validate the required input fields.
    for index, item in enumerate(data, start=1):
        if "id" not in item or "text" not in item:
            raise ValueError(
                f"Record {index} must contain 'id' and 'text'."
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize the selected model.
    if args.model == "gliner":
        model = GLiNERRunner()
        model_name = "gliner"
    else:
        model = LLMRunner(model_name=args.model_name)
        model_name = args.model_name.replace(":", "_").replace("/", "_")

    # Warm up using up to three records.
    # Warm-up calls are excluded from measured latency.
    print("Warming up model...")

    for item in data[:3]:
        try:
            model.predict(item["text"])
        except Exception as exc:
            print(f"Warm-up warning for {item['id']}: {exc}")

    print("Starting timed predictions...")

    predictions = []
    latencies = []
    prediction_errors = 0

    for index, item in enumerate(data, start=1):
        start_time = time.perf_counter()

        try:
            raw = model.predict(item["text"])
        except Exception as exc:
            print(
                f"Prediction error for {item['id']}: {exc}"
            )
            raw = {"entities": []}
            prediction_errors += 1

        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)

        # Apply the shared postprocessing rules.
        entities = postprocess(raw, item["text"])

        predictions.append({
            "id": item["id"],
            "entities": entities,
        })

        if index % 10 == 0 or index == len(data):
            print(
                f"Processed {index}/{len(data)} snippets"
            )

    # Use the dataset filename to keep dev/test results separate.
    dataset_name = data_path.stem

    prediction_path = (
        RESULTS_DIR
        / f"predictions_{model_name}_{dataset_name}.jsonl"
    )

    timing_path = (
        RESULTS_DIR
        / f"timings_{model_name}_{dataset_name}.json"
    )

    write_jsonl(prediction_path, predictions)

    summary = {
        "model": model_name,
        "dataset": str(data_path),
        "snippet_count": len(data),
        "prediction_errors": prediction_errors,
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": calculate_p95(latencies),
        "latencies_seconds": latencies,
    }

    with open(timing_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nCompleted.")
    print(f"Predictions: {prediction_path}")
    print(f"Timings: {timing_path}")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key != "latencies_seconds"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
