# Deterministic Affordability Agent 
**Hybrid LLM Financial Pipeline with Zero Hallucination**

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square&logo=python&logoColor=white)
![Claude 4.6 Sonnet](https://img.shields.io/badge/Claude%204.6%20Sonnet-Anthropic-8A2BE2?style=flat-square)
![Pydantic](https://img.shields.io/badge/Pydantic-Data%20Validation-e92063?style=flat-square)
![HackerRank](https://img.shields.io/badge/HackerRank-Bronze%20Medal-CD7F32?style=flat-square&logo=hackerrank&logoColor=white)

![HackerRank Orchestrate Certificate](./assets/certificate.png)
*HackerRank Orchestrate Hackathon (September 2026) — Bronze Medal (Ranked 422 / 3,062)*

---

## Overview

The **Deterministic Affordability Agent** is an AI-powered financial router that evaluates purchase requests and autonomously generates personalized payment schedules (full payment, installments, partial payments, or wait). 

To solve the critical enterprise risk of LLM financial hallucination, this system utilizes a **hybrid deterministic-plus-LLM architecture**. The language model is strictly prohibited from inventing money, choosing dates, or calculating schedules. Instead, all mathematical ledger operations, 90-day cash flow forecasts, and state routing are executed securely in a deterministic Python kernel.

---

## Architectural Approach

**Claude Sonnet 4.6** is strictly constrained to two specific tasks:
1. **Multimodal Vision Extraction:** Extracting payable amounts from linked receipt/statement images when ledger data is missing.
2. **NLP Explanation:** Rewriting an already-validated, Python-generated payment plan into a human-readable `decision_explanation` citing the user's financial priorities.

Everything that impacts the user's cash state—balances, FX conversions, recurring transaction detection, spending cuts, and payment scheduling—is computed in Python using strict `Decimal` arithmetic. Forced tool-calling and a post-generation number-lock check guarantee that the LLM cannot introduce a date or figure that is not already mathematically proven by the deterministic plan.

---

## Pipeline Flow

```text
dataset/  →  evidence  →  ledger  →  forecast  →  candidates
                                                      ↓
output.csv  ←  explanation (Claude)  ←  rank  ←  validate
```

1. **Load** structured files from `dataset/` only (`requests.csv`, profiles, events, FX, payment options, messages, images). Sample output columns are not used as labels for evaluation requests.
2. **Evidence.** Resolve blank event amounts from `dataset/media/images/<image_id>.png`. Apply typed message/image facts (cancellation, date shift, amount amendment) without executing prompt-injection text.
3. **Ledger.** Reconstruct cash state: reserve pending *debits*, ignore pending credits until they settle, count confirmed salary on its settlement date, detect recurrence only when history supports it, convert foreign-currency events with the dated row in `exchange_rates.csv`.
4. **Forecast.** Project ~90 days conservatively so the balance never falls below `minimum_balance_to_keep` after essential spending. Compute `amount_safe_to_pay` on `request_date` and the earliest date a full payment is safe.
5. **Planning.** Generate full-payment, partial-payment, installment, wait, and (if the user allows) spending-change candidates. Installments must match a supplied seller option and `max_installment_months`. Partial payment is exactly two installments that sum to `requested_amount`.
6. **Validation.** An independent checker re-simulates the plan and enforces bounds, chronology, preferences, protected categories, and deadline rules. Invalid candidates are dropped before ranking.
7. **Ranking.** Prefer plans that complete by the deadline, avoid spending changes, minimize total cost, start earlier, and use fewer payments.
8. **Explanation.** Claude Sonnet 4.6 writes `decision_explanation` from the frozen plan. Failures fall back to a deterministic sentence that still cites priorities and copies plan values.
9. **Write** one `output.csv` row per evaluation request, then `evaluation/usage_report.md` for the same run.

---

## Setup

Python 3.11+ is sufficient. From the repository root (the directory that contains `dataset/` and `code/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env`. The key is read entirely at runtime and is never exposed to output files or logs. If the key is unset, the agent gracefully degrades to offline vision fallbacks and deterministic explanations. Optional overrides:

```text
ANTHROPIC_VISION_MODEL=claude-sonnet-4-6
ANTHROPIC_EXPLANATION_MODEL=claude-sonnet-4-6
```

---

## Execution

```bash
python3 code/main.py
```

This command:

- reads every row in `dataset/requests.csv`
- writes `output.csv` at the repository root (exact columns, one row per `request_id`)
- overwrites `code/evaluation/usage_report.md` with token and cost totals for that full-dataset run

Optional:

```bash
python3 code/main.py --split calibration   # request_01..10
python3 code/main.py --split holdout       # request_11..25 (no labels used)
python3 code/main.py --limit 5             # debug; does not overwrite the submission usage report
```

---

## Output contract

`output.csv` columns, in this order:

```text
request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation
```

- `0 <= amount_safe_to_pay <= requested_amount`
- `affordability_status`: `affordable_now` | `affordable_with_plan` | `affordable_later` | `not_affordable`
- `recommended_payment_method`: `full_payment` | `partial_payment` | `installments` | `wait` | `not_recommended`
- `payment_plan`: chronological `YYYY-MM-DD:amount` entries joined by `|`, or `none`
- `spending_changes_needed`: up to three `stop:<event_id>` / `reduce_to:<event_id>:<amount>` actions on flexible, non-protected events, or `none`

---

## Layout

```text
.
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── .env.example              # Placeholder only; never commit a real key
├── code/
│   ├── main.py               # Pipeline orchestrator
│   ├── domain.py             # Frozen Pydantic models and decimal formatting
│   ├── tools/                # Ledger, forecast, planning, validation logic
│   └── evaluation/
│       └── usage_report.md   # Automated token and USD cost tracking
│       └── full_audit.py     # Statistical & numerical audit suite
│       └── calibrate.py      # Calibration pipeline for threshold tuning
├── dataset/                  # Structured inputs and media files
└── output.csv                # Pipeline predictions
```

---

## Token usage

The full evaluation run that produced `output.csv` is summarized in `evaluation/usage_report.md` (also at `code/evaluation/usage_report.md`). It reports provider, model name, call counts, input/output/total tokens, averages per request, and estimated USD cost. Money math is not billed as tokens.

---

## Design constraints honored

- Runnable from the terminal (`python3 code/main.py`)
- Reads only the provided `dataset/` files
- One prediction per `request_id` in `dataset/requests.csv`
- No organizer-only files and no hardcoded evaluation labels
- Deterministic kernel; LLM output is constrained and verified
- Secrets from environment variables only

---

## License
This project is licensed under the MIT License.
