# LLM Fine-Tuning for Reliable Tool Calling

**Fine-tuning Qwen2.5-1.5B-Instruct to emit valid JSON tool calls, and comparing LoRA, QLoRA, DoRA and AdaLoRA on a single free T4 GPU.**

A small base model usually answers a function-calling request in prose or a code block rather than a clean tool call. This project fine-tunes it so every response is a parseable `{"name": ..., "arguments": {...}}` object, then measures which parameter-efficient method gets there at the lowest cost.

## Key results

Evaluated on 200 held-out prompts, with no JSON instruction in the prompt:

| Model | Valid JSON | Correct tool | Exact match (name + arguments) |
|---|---|---|---|
| Base Qwen2.5-1.5B-Instruct | **0%** | 0% | 0% |
| LoRA (r=8) | **100%** | **100%** | **94%** (188/200) |

- Valid JSON tool calls rose from **0% to 100%**, with **100% correct tool selection** and **94% exact-argument match**.
- All four adapter methods reached the same accuracy, so the choice came down to cost: **LoRA r=8** trains **0.14%** of the parameters in about **6 minutes** on a T4.
- **QLoRA cut weight memory by 62%** (2.88 GB to 1.09 GB) but trained **39% slower** (8.2 min vs 5.9 min).
- All 12 remaining LoRA errors were **wrong argument values, never a wrong tool**.

## Method comparison

Same data, seed, learning rate, batch size, target modules and 1 epoch (900 steps) for every run.

| Method | Trainable params | Weights (GB) | Peak GPU (GB) | Train time (min) | Valid JSON | Correct tool | Exact match |
|---|---|---|---|---|---|---|---|
| LoRA r=8 | 2.18M (0.141%) | 2.88 | 4.70 | 5.9 | 200/200 | 200/200 | 188/200 |
| QLoRA r=8 (4-bit NF4) | 2.18M (0.141%) | 1.09 | 3.41 | 8.2 | 200/200 | 200/200 | 189/200 |
| DoRA r=8 | 2.28M (0.147%) | 2.89 | 4.70 | 8.6 | 200/200 | 200/200 | 188/200 |
| AdaLoRA r=8 | 3.27M (0.211%) | 2.89 | 4.76 | 9.0 | 200/200 | 200/200 | 189/200 |

### LoRA rank sweep

| Rank | Trainable params | Train time (min) | Correct tool | Exact match |
|---|---|---|---|---|
| 4 | 1.09M (0.071%) | 5.8 | 200/200 | 187/200 |
| 8 | 2.18M (0.141%) | 5.9 | 200/200 | 188/200 |
| 16 | 4.36M (0.282%) | 5.8 | 200/200 | 190/200 |
| 32 | 8.72M (0.561%) | 5.8 | 200/200 | 191/200 |

Tool accuracy is 100% at every rank, and exact match differs by only 4 prompts between r=4 and r=32, so a small rank is enough for this task.

### Prompt-only baseline

As an extra row, the untouched base model was also given an explicit "reply with ONLY a JSON object" instruction in the prompt. It scored 200/200 valid JSON, 200/200 correct tool and 190/200 exact match. Fine-tuning therefore matches this prompt-based result, but with the behaviour built into the weights instead of depending on prompt wording. It also keeps the prompt shorter.

## Setup

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Data:** `glaiveai/glaive-function-calling-v2`. Chats whose first assistant turn is a function call were parsed into clean JSON targets and capped at 700 tokens, giving 2,000 examples split into **1,800 train / 200 test** (seed 42).
- **Training:** 1 epoch, batch size 2, learning rate 2e-4, gradient checkpointing, loss on the answer tokens only.
- **Adapters:** targets `q_proj`, `k_proj`, `v_proj`, `o_proj`; built with Hugging Face PEFT, with bitsandbytes for the 4-bit QLoRA run.
- **Metrics:**
  - *Valid JSON:* the output parses as JSON.
  - *Correct tool:* the function name matches the reference.
  - *Exact match:* name and all arguments match the reference.

## Error analysis

The 12 remaining LoRA r=8 errors are all argument-level, for example:
- A different argument key (`"actors": ["Tom Hanks"]` instead of `"keyword": "Tom Hanks"`).
- A hallucinated or reformatted value (dates for a relative request like "25th of next month", `"electronics"` instead of `"Electronics"`).
- An omitted optional argument (`max_prep_time`).

## Run it yourself

The notebook `LM_tool_calling.ipynb` is written for Google Colab.

1. Open it in Colab and choose **Runtime > Change runtime type > T4 GPU**.
2. Run the cells top to bottom. Results are saved to `results.json` on Google Drive after every run, so finished runs are skipped if the session disconnects.
3. Run the five comparison rows (base, LoRA, QLoRA, DoRA, AdaLoRA), then the rank sweep for the winning method.

Dependencies: `peft`, `bitsandbytes`, `accelerate`, `transformers`, `datasets`, `torch`, `pandas`, `matplotlib`.

## Limitations

- One epoch and one seed, on a 200-prompt test set, so differences of one or two prompts between methods are within noise.
- Only the first assistant turn of each chat is used, so multi-turn tool use is not covered.
- Exact match compares against a single reference call, so it can penalise valid alternatives, such as a different but reasonable argument key.
- Generation times in the notebook (`gen_seconds_200`) were recorded as a side measurement, not a controlled inference benchmark.
