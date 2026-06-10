# A/B Test Decision Workbench

Experiment readout for product and growth decisions: hypothesis, sample health, primary metric testing, guardrail checks, power/MDE, segment heterogeneity, and a ship / ramp / keep-running decision memo.

Live demo: https://ab-test-analyzer.streamlit.app/

Project page: https://www.josephjwang.com/demos/ab-test-analyzer

## Business Question

For an e-commerce checkout or landing-page experiment, should the new experience ship?

The workbench is designed to answer that question the way a product data scientist would:

1. Confirm assignment health and sample ratio mismatch risk.
2. Test the primary conversion metric.
3. Check guardrail metrics such as revenue, sessions, page-load time, and bounce.
4. Compare current sample size against MDE and power.
5. Inspect segment-level treatment effects with multiple-testing correction.
6. Produce a concise decision memo.

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
- Upload any user-level experiment CSV with `user_id`, `variant`, and `converted`.
- Sample ratio mismatch check using chi-square test.
- Primary conversion test using a two-proportion z-test and 95% confidence interval.
- Guardrail metric tests using Welch t-tests for numeric columns.
- Power and MDE panel for experiment planning.
- Segment heterogeneity readout with Benjamini-Hochberg adjusted p-values.
- Structured decision memo for interview and stakeholder storytelling.
- Light Streamlit theme for readable public demo presentation.

## Input Format

Required columns:

| Column | Type | Notes |
|---|---|---|
| `user_id` | int / str | One row per experimental unit |
| `variant` | str | `control` or `treatment` |
| `converted` | int | Primary metric, 0 or 1 |

Optional columns:

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

The app works without an API key. The built-in sample dataset is generated in the app, and users can download it as a CSV.

## Portfolio Story

This project complements the two uplift projects:

- **Promotion uplift**: owned-channel CRM experiment and targeting policy.
- **Advertising uplift**: paid-media incrementality and budget allocation.
- **A/B test workbench**: product experiment decision process with guardrails and power.

Together, they show an experiment analytics workflow across CRM, paid ads, and product experience optimization.

---

[josephjwang.com](https://josephjwang.com) · [github.com/josephwang-ds](https://github.com/josephwang-ds)
