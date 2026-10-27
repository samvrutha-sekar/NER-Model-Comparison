# NER Model Comparison Report

Generated (UTC): 2026-10-27T18:32:56.996575+00:00

## Method

Models: GLiNER, Qwen3 4B, and Qwen3 8B.
All models use the project's shared postprocessing and unchanged score.py.
Development data is used for development-stage comparison; test data is held out.
Latency is measured by the runner and excludes warm-up calls.

## Development results

| Model | Micro precision | Micro recall | Micro F1 | Median latency (s) | P95 latency (s) |
|---|---:|---:|---:|---:|---:|
| gliner | 0.656 | 0.875 | 0.750 | 0.146 | 0.157 |
| qwen3_4b | 1.000 | 0.958 | 0.979 | 5.654 | 9.344 |
| qwen3_8b | 0.923 | 1.000 | 0.960 | 7.047 | 9.152 |

## Held-out test results

| Model | Micro precision | Micro recall | Micro F1 | Median latency (s) | P95 latency (s) |
|---|---:|---:|---:|---:|---:|
| gliner | 0.591 | 0.750 | 0.661 | 0.152 | 0.189 |
| qwen3_4b | 0.842 | 0.889 | 0.865 | 5.694 | 9.477 |
| qwen3_8b | 0.883 | 0.981 | 0.930 | 6.871 | 10.191 |

## Interpretation and limitations

- Use held-out test results for final performance claims.
- The development dataset contains only a small number of snippets; results may be unstable.
- Latency depends on hardware, model warm-up, and system load.
- Peak memory was not measured by this script.
- Monetary cost was not calculated because deployment prices and a deployment setup were not provided.
- Error analysis CSV files list false positives and false negatives by model and snippet.
- Review dataset quality, label definitions, and model outputs before submission.

## Generated artifacts

- `dev_comparison.csv` and `dev_comparison.json`
- `test_comparison.csv` and `test_comparison.json`
- `dev_error_analysis.csv` and `test_error_analysis.csv`
- `frozen_config.json`
- `test_run_manifest.json`

## AI-use disclosure

Complete this section accurately according to the assistance and tools used in the project.
