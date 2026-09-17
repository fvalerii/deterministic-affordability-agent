# Token usage report

This file is the single token-usage artifact required in `code.zip` as `evaluation/usage_report.md`. It contains **no API keys, credentials, or environment values**.

## Run that produced this report

- Output artifact: `output.csv`
- Dataset split: `eval`
- UTC start: 2026-09-13T10:22:51Z
- Evaluation requests processed: 250
- Local SHA-256 cache hits (no API call): 0

## Overall totals

- Provider: Anthropic
- Model names: claude-sonnet-4-6
- Model calls: 411
- Input tokens: 467918
- Output tokens: 39044
- Total tokens: 506962
- Average tokens per request: 2027.85
- Estimated total cost (USD): 1.989414
- Estimated cost per request (USD): 0.007958

## Per-model totals

| provider | model | calls | input tokens | output tokens | total tokens | purposes | estimated cost (USD) |
|---|---|---:|---:|---:|---:|---|---:|
| Anthropic | `claude-sonnet-4-6` | 411 | 467918 | 39044 | 506962 | explanation=400, vision=11 | 1.989414 |

Pricing used for `claude-sonnet-4-6`: $3.00/M input, $15.00/M output (Anthropic published list prices, global standard routing).

**Overall** — calls 411, tokens 506962, estimated cost $1.989414.

## Notes

- Money math is done by deterministic Python tools and is not billed as tokens.
- Vision (receipt OCR) and explanation synthesis are the only Anthropic calls.
- Average tokens per request uses evaluation requests in `output.csv`, not model-call count.
- Re-running the same images or plans hits an in-process SHA-256 cache and does not add API tokens.
