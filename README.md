\# NER model comparison: GLiNER vs Qwen3 4B vs Qwen3 8B



Compares one specialist NER model and two local LLMs on the provided 4-label NER task (PERSON, ORG, LOCATION, EVENT) for quality, latency, memory and estimated cost. All models run locally on CPU.





\## Results (test set, 50 snippets, 108 reference entities)



| Model | F1 | PERSON | ORG | LOCATION | EVENT | Median latency | p95 latency | Peak RSS | Cost / 1k snippets |

|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| GLiNER medium v2.1 | 0.661 | 0.667 | 0.343 | 0.879 | 0.491 | 0.152 s | 0.189 s | 2.20 GiB | $0.015 |

| Qwen3 4B (Q4\_K\_M) | 0.865 | 0.846 | 0.880 | 0.929 | 0.722 | 5.694 s | 9.477 s | 2.79 GiB | $0.565 |

| Qwen3 8B (Q4\_K\_M) | 0.930 | 1.000 | 0.980 | 0.965 | 0.700 | 6.871 s | 10.191 s |     5.6 GB     | $0.681 |



F1 is micro-averaged and comes from the unmodified `score.py`. Per-label columns are F1. Precision and recall, with 95% Wilson intervals:



| Model | TP / FP / FN | Precision \[95% CI] | Recall \[95% CI] |

|---|---|---|---|

| GLiNER | 81 / 56 / 27 | 0.591 \[0.508, 0.670] | 0.750 \[0.661, 0.822] |

| Qwen3 4B | 96 / 18 / 12 | 0.842 \[0.764, 0.898] | 0.889 \[0.816, 0.935] |

| Qwen3 8B | 106 / 14 / 2 | 0.883 \[0.814, 0.929] | 0.981 \[0.935, 0.995] |



These intervals treat entities as independent trials. Entities from the same snippet are not fully independent, so the intervals are somewhat optimistic.



\## Hardware and software



| Item | Value |

|---|---|

| Laptop | Acer Aspire Go 14 AG14-71M |

| CPU | Intel Core Ultra 5 125H (14 cores / 18 threads) |

| RAM | 16 GB |

| GPU | Integrated Intel Graphics. Not used: all inference ran on CPU |

| OS | Windows 11 Home Single Language, 10.0.26200 |

| Python | 3.12.0 |

| PyTorch | 2.14.1 (CPU build) |

| gliner | 0.2.29 |

| ollama (Python client) | 0.6.3 |

| Ollama server |  0.40.11 |



\## Models and settings



| Model | Identifier | Size / precision | Settings |

|---|---|---|---|

| GLiNER | `urchade/gliner\_medium-v2.1` | F32 weights, no quantisation | threshold 0.4, `flat\_ner=True`. Labels passed to the model: \["PERSON", "ORG", "LOCATION", "EVENT"] |

| Qwen3 4B | `qwen3:4b` (Ollama) | 4.0B params, Q4\_K\_M, 2.5 GB | temperature 0, thinking disabled, JSON-schema output, `num\_ctx` 2048, max 256 output tokens |

| Qwen3 8B | `qwen3:8b` (Ollama) | 8.2B params, Q4\_K\_M, 5.2 GB | same as 4B |





\## Repository layout



```

data/            dev.jsonl, test.jsonl (as provided)

score.py         provided scorer, unmodified

Harness/         runners, shared postprocess.py, run script

Results/         predictions\_\*.jsonl, timing JSON, score outputs, error CSVs,

&#x20;                memory\_measurements.csv, frozen\_config.json (code + data SHA-256 hashes)

```



\## Setup (Windows, PowerShell)



```powershell

py -3.12 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt          # CPU PyTorch: see the comment in requirements.txt

ollama pull qwen3:4b

ollama pull qwen3:8b

```



\## Reproduce



```powershell

\# 1. Development runs (all tuning happened here)

python Harness/run.py --model gliner --split dev

\# 2. Final test runs (one run per model, frozen config)

python Harness/run.py --model gliner   --split test

python Harness/run.py --model qwen3\_4b --split test

python Harness/run.py --model qwen3\_8b --split test

\# 3. Scoring

python score.py data/test.jsonl Results/predictions\_gliner.jsonl --errors

\# 4. Memory (separate run on dev; stop other Ollama models first)

ollama stop qwen3:4b; ollama ps

python Harness/measure\_memory.py --model qwen3\_8b --split dev

```



\## Method



\- \*\*Dev/test discipline:\*\* prompts, the GLiNER threshold and the post-processing rules were developed on `dev.jsonl` only. The configuration was frozen, with hashes in `Results/frozen\_config.json`, before one test run per model. Test errors were inspected only after that run, and nothing was changed afterwards.

\- \*\*Latency:\*\* `time.perf\_counter()` around the model call only. Model load and 3 warm-up calls are excluded, and post-processing runs outside the timer. Snippets were processed sequentially, one request at a time. 

\- \*\*Memory:\*\* peak resident set size (RSS), sampled every 0.2 s on a separate 10-snippet dev run. For the Qwen models this is the sum over Ollama processes, with only the model under test loaded. Sampling can miss short spikes.

\- \*\*Post-processing:\*\* the same rules for all models (`Harness/postprocess.py`):

&#x20; 1. Keep only the 4 labels.

&#x20; 2. Strip titles (Mr, Mrs, Ms, Miss, Dr, Minister).

&#x20; 3. Strip possessives.

&#x20; 4. Drop spans not found in the input (case-insensitive).

&#x20; 5. Dedupe on (text, label), case-insensitive.

&#x20; 6. Prefer the longer span when spans overlap.

\- \*\*LLM output handling:\*\* structured JSON output, with a fallback parser. Parse failures are logged and scored as no entities. There were 0 parse failures on test.



\## Cost assumption



\- \*\*Instance:\*\* AWS EC2 c7i.2xlarge (8 vCPU, 16 GiB), Linux on-demand, ap-south-1 (Mumbai), $0.357/hour.

&#x20; - Source: (https://aws.amazon.com/ec2/pricing/)

&#x20; - Why this instance: it's the smallest general CPU instance with 16 GiB, matching the laptop's RAM, and the reason for choosing Mumbai.

\- \*\*Formula:\*\* cost per 1k snippets = $0.357 / 3600 × median seconds × 1000.

\- \*\*What it leaves out:\*\*

&#x20; - A c7i.2xlarge has 4 physical cores against the laptop's 14, so cloud latency is likely \*higher\* than measured here.

&#x20; - No batching or concurrency.

&#x20; - A GPU instance would change the LLM numbers substantially.

&#x20; - Model load, idle time, storage, transfer and engineering effort are not counted.



\## Assumptions



\- "Snippet latency" means model-call time only, excluding post-processing (which takes milliseconds).

\- GLiNER and the Qwen models use the same post-processing, so the scores reflect model plus rules, not raw model output.



\## AI-assistant disclosure



I used Claude to review my harness code, help debug, compute the derived statistics, and help structure and edit the README . I ran all experiments and checked every reported number against the saved outputs.



