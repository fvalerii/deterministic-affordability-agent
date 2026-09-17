# Buy or Wait?

An AI-powered affordability agent for the HackerRank Orchestrate challenge (September 2026). For every purchase or payment request it decides whether to pay in full, pay in two parts, use a seller installment option, wait, or not proceed — without letting the language model invent money.

## Approach

This is a **hybrid deterministic-plus-LLM pipeline**.

**Claude Sonnet 4.6** (`claude-sonnet-4-6`, Anthropic) is used for two tasks only:

1. **Vision** — extract a positive payable amount from a linked receipt or statement when `financial_events.csv` leaves `amount` blank. A blank amount is never treated as zero.
2. **Explanation** — rewrite an already-validated plan into a short `decision_explanation` that cites the user's `financial_priorities`.

Everything that affects cash — balances, FX, recurrence, forecasts, payment schedules, spending cuts, and the recommended method — is computed in Python with `Decimal` arithmetic. The model never chooses an amount, a date, a payment method, or a spending change. Forced tool-calling plus a number-lock check reject any explanation that introduces a date or figure that is not already on the plan.

Messages and images are treated as untrusted evidence. They may confirm, amend, delay, or cancel a fact, but embedded instructions (including advance-fee bait) never override the challenge rules.

## Pipeline

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

## Setup

Python 3.11+ is sufficient. From the repository root (the directory that contains `dataset/` and `code/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env` or in the environment. The key is read at runtime and is never written to `output.csv`, logs, or `evaluation/usage_report.md`. Optional overrides:

```text
ANTHROPIC_VISION_MODEL=claude-sonnet-4-6
ANTHROPIC_EXPLANATION_MODEL=claude-sonnet-4-6
```

If the key is unset, the agent still produces a valid `output.csv` using offline vision fallbacks and deterministic explanations. The submitted `output.csv` was generated with Claude Sonnet 4.6 enabled.

## Execution

```bash
python3 code/main.py
```

That command:

- reads every row in `dataset/requests.csv`
- writes `output.csv` at the repository root (exact columns, one row per `request_id`)
- overwrites `code/evaluation/usage_report.md` with token and cost totals for that full-dataset run

Optional:

```bash
python3 code/main.py --split calibration   # request_01..10
python3 code/main.py --split holdout       # request_11..25 (no labels used)
python3 code/main.py --limit 5             # debug; does not overwrite the submission usage report
```

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

## Layout

```text
.
├── README.md                 # this file
├── requirements.txt
├── .env.example              # placeholder only; never commit a real key
├── code/
│   ├── main.py               # entry point
│   ├── domain.py             # frozen Pydantic models and money formatting
│   ├── tools/                # ledger, forecast, planning, validation, LLM adapters
│   └── evaluation/
│       └── usage_report.md   # token/cost report for the 250-row run
├── dataset/                  # provided inputs (not modified)
└── output.csv                # 250 predictions
```

## Token usage

The full evaluation run that produced `output.csv` is summarized in `evaluation/usage_report.md` (also at `code/evaluation/usage_report.md`). It reports provider, model name, call counts, input/output/total tokens, averages per request, and estimated USD cost. Money math is not billed as tokens.

## Design constraints honored

- Runnable from the terminal (`python3 code/main.py`)
- Reads only the provided `dataset/` files
- One prediction per `request_id` in `dataset/requests.csv`
- No organizer-only files and no hardcoded evaluation labels
- Deterministic kernel; LLM output is constrained and verified
- Secrets from environment variables only
