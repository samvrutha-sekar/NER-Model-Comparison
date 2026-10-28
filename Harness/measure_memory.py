

import csv
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "memory_dev.jsonl"
OUTPUT = ROOT / "Results" / "memory_measurements.csv"


def process_tree_rss(pid):
    """Return RSS memory for a process and its child processes."""
    try:
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)
    except psutil.Error:
        return 0

    total = 0
    for proc in processes:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            pass

    return total


def ollama_rss():
    """Return RSS for Ollama server and model-runner processes."""
    total = 0

    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info["name"] or "").lower()
            cmdline = " ".join(proc.info["cmdline"] or []).lower()

            # Include the Ollama server and its model runner.
            if (
                "ollama" in name
                or "ollama" in cmdline
                or "llama" in name
                or "llama" in cmdline
            ):
                total += proc.memory_info().rss

        except (psutil.Error, OSError):
            continue

    return total


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python .\\Harness\\measure_memory.py gliner")
        print("  python .\\Harness\\measure_memory.py qwen3:4b")
        print("  python .\\Harness\\measure_memory.py qwen3:8b")
        return

    model = sys.argv[1].lower()

    if model not in {"gliner", "qwen3:4b", "qwen3:8b"}:
        print("Unknown model. Choose gliner, qwen3:4b, or qwen3:8b.")
        return

    if not DATA.exists():
        print(f"Dataset not found: {DATA}")
        print("Create memory_dev.jsonl as a copy of dev.jsonl first.")
        return

    if model == "gliner":
        command = [
            sys.executable,
            str(ROOT / "Harness" / "run.py"),
            "--model", "gliner",
            "--data", str(DATA),
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "Harness" / "run.py"),
            "--model", "qwen3",
            "--model-name", model,
            "--data", str(DATA),
        ]

    print(f"Model: {model}")
    print(f"Dataset: {DATA.name}")
    print("Sampling process memory every 0.2 seconds.")
    print("Do not run another model during this measurement.")

    process = subprocess.Popen(command, cwd=ROOT)
    peak_bytes = 0

    try:
        while process.poll() is None:
            if model == "gliner":
                current_bytes = process_tree_rss(process.pid)
            else:
                current_bytes = ollama_rss()

            peak_bytes = max(peak_bytes, current_bytes)
            time.sleep(0.2)

    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        print("Measurement interrupted; result not saved.")
        return

    if process.returncode != 0:
        print(f"Runner failed with exit code {process.returncode}.")
        print("Check the error above; no memory result was saved.")
        return

    peak_mib = round(peak_bytes / (1024 ** 2), 2)

    print(f"\nPeak sampled process RSS: {peak_mib} MiB")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT.exists()

    with OUTPUT.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "model",
                "dataset",
                "peak_sampled_process_rss_mib",
                "sampling_interval_seconds",
            ])

        writer.writerow([
            model,
            DATA.name,
            peak_mib,
            0.2,
        ])

    print(f"Saved measurement to: {OUTPUT}")
    print(
        "Note: this is sampled process RSS, not total system RAM "
        "or guaranteed absolute peak memory."
    )


if __name__ == "__main__":
    main()
