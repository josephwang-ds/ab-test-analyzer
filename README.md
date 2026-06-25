# A/B Test Decision Workbench

Experiment readout for product and growth decisions: hypothesis, sample health, primary metric testing, guardrail checks, power/MDE, segment heterogeneity, and a ship / ramp / keep-running decision memo.

Live demo: https://ab-test-analyzer.streamlit.app/

Project page: https://www.josephjwang.com/demos/ab-test-analyzer

## Working Scenario

**Product under test — Trailhead single-page checkout redesign.** Trailhead is a mid-market DTC outdoor & apparel retailer (~$120M annual GMV, ~58% of sessions on mobile). Over two quarters mobile traffic share climbed to 58%, but mobile checkout conversion (8.0%) kept trailing desktop (10.0%).

- **Business case (why run it):** before committing engineering to a full rollout, the team wants to know whether a redesigned checkout closes the mobile gap.
- **What changes:** treatment replaces the legacy 3-step checkout (cart → shipping → payment) with a single-page checkout — address autofill, express wallets (Apple Pay / Shop Pay), fewer required fields. Control keeps the 3-step flow. Users are randomized 50/50 at first checkout entry.
- **Where the idea came from:** (1) funnel analysis showing the largest drop-off at the shipping/payment step; (2) session replays of mobile users abandoning the long form; (3) support tickets and post-purchase verbatims citing "checkout is too long."
- **Risks to watch (→ guardrails & segments):** express checkout may cut order value (revenue per user); wallet SDKs may slow the page (page-load time); a confusing layout may raise bounce; fewer steps may reduce sessions; partial-device rendering can skew assignment (SRM); the effect may differ by device/channel/country (segment heterogeneity).

The decision question — *should the new checkout ship?* — is answered the way a product data scientist would:

1. Confirm assignment health and sample ratio mismatch risk.
2. Test the primary conversion metric.
3. Check guardrail metrics such as revenue, sessions, page-load time, and bounce.
4. Compare current sample size against MDE and power.
5. Inspect segment-level treatment effects with multiple-testing correction.
6. Produce a concise decision memo.

## Metrics & Goals

| Metric | Role | Definition (per arm) | Test |
|---|---|---|---|
| **Conversion rate** | Primary (higher better) | `converted_users / total_users` — user-level, each user counted once; converted = placed ≥1 order in the window | Two-proportion z-test, unpooled 95% CI, Cohen's h |
| Revenue per user | Guardrail (higher better) | `sum(revenue) / total_users` incl. $0 non-converters (ARPU) | Mann-Whitney U (zero-inflated) |
| Sessions per user | Guardrail (higher better) | `sum(sessions) / total_users` | Welch t-test |
| Page load (ms) | Guardrail (lower better) | `mean(page_load_ms)` | Welch t-test |
| Bounce rate | Guardrail (lower better) | `bounced_users / total_users` | Welch t-test |

**Decision rule:** SHIP only when conversion rate is significantly positive (p < α) with no material adverse guardrail movement; otherwise RAMP WITH CAUTION, KEEP RUNNING (effect below current MDE), or DO NOT SHIP.

## Why This Project Matters

Basic A/B test demos often stop at "p-value < 0.05." Real experiment decisions need more structure:

- **Hypothesis clarity**: what behavior should change and why?
- **Metric hierarchy**: what is the primary metric, and what are the guardrails?
- **Experiment health**: is the treatment/control split trustworthy?
- **Power and MDE**: is a non-significant result truly neutral, or just underpowered?
- **Heterogeneous effects**: does the treatment help one segment and hurt another?
- **Decision framing**: ship, ramp, keep running, or redesign.

## Features

- Built-in e-commerce sample experiment with `device`, `channel`, `country`, `converted`, `revenue`, `sessions`, `page_load_ms`, and `bounce`.
- **Smart CSV column mapping**: upload any column names — the app auto-detects `user_id` / `variant` / `converted` from headers and values (e.g. `ab_group` → variant, `purchased`/`yes,no` → converted, `A,B` → control/treatment) and lets you confirm or override. Optional LLM-assisted mapping via DeepSeek, Qwen, or OpenAI (resolution order DeepSeek → Qwen → OpenAI, or force with `LLM_PROVIDER`), with automatic fallback to the rule-based mapper.
- Sample ratio mismatch check using chi-square test.
- Primary conversion test using a two-proportion z-test and 95% confidence interval.
- Guardrail metric tests using Welch t-tests for numeric columns.
- Power and MDE panel for experiment planning.
- Segment heterogeneity readout with Benjamini-Hochberg adjusted p-values.
- Structured decision memo for interview and stakeholder storytelling.
- Light Streamlit theme for readable public demo presentation.

## Input Format

No fixed template is required — column names are auto-mapped. The canonical fields are:

| Canonical field | Type | Auto-detected from (examples) |
|---|---|---|
| `user_id` | int / str | `user`, `uid`, `visitor_id`, or a near-unique column |
| `variant` | str | `group`, `arm`, `bucket`, values like `control/treatment` or `A/B` |
| `converted` | int | `purchased`, `order`, values like `0/1`, `true/false`, `yes/no` |

Optional columns are detected automatically:

- numeric guardrails: `revenue`, `sessions`, `page_load_ms`, `bounce`, etc.
- categorical segments: `device`, `channel`, `country`, `member_tier`, etc.

## Methods

| Step | Method |
|---|---|
| Sample ratio mismatch | Chi-square goodness-of-fit test |
| Primary conversion | Two-proportion z-test |
| Confidence interval | Unpooled 95% CI for treatment-control difference |
| Numeric guardrails | Welch t-test |
| Power / MDE | Normal approximation for two independent proportions |
| Segment correction | Benjamini-Hochberg multiple-testing correction |

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app works without an API key — rule-based column mapping is the default and the built-in sample dataset is generated in-app. To enable optional LLM-assisted column mapping, set one of `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY` (Qwen), or `OPENAI_API_KEY` (in `.env`, environment, or Streamlit secrets); without any key the app silently falls back to the rule-based mapper. All three are reached through the OpenAI-compatible SDK, and column mapping is a trivial task, so the cheapest chat model from any provider is sufficient.

## Portfolio Story

This project complements the two uplift projects:

- **Promotion uplift**: owned-channel CRM experiment and targeting policy.
- **Advertising uplift**: paid-media incrementality and budget allocation.
- **A/B test workbench**: product experiment decision process with guardrails and power.

Together, they show an experiment analytics workflow across CRM, paid ads, and product experience optimization.

---

[josephjwang.com](https://josephjwang.com) · [github.com/josephwang-ds](https://github.com/josephwang-ds)
