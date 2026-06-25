"""
A/B Test Decision Workbench

Upload user-level experiment data or use the built-in e-commerce sample.
The app checks experiment design, primary metric significance, guardrails,
power/MDE, segment heterogeneity, and a ship/no-ship decision memo.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


st.set_page_config(
    page_title="A/B Test Decision Workbench",
    page_icon="AB",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
STREAMLIT_URL = "https://ab-test-analyzer.streamlit.app/"
WEBSITE_URL = "https://www.josephjwang.com/demos/ab-test-analyzer"
GITHUB_URL = "https://github.com/josephwang-ds/ab-test-analyzer"


# ── Concrete business scenario (illustrative DTC e-commerce experiment) ──────
SCENARIO = {
    "en": {
        "product": "Trailhead — single-page checkout redesign",
        "company": "Trailhead, a mid-market direct-to-consumer outdoor & apparel retailer (~$120M annual GMV, ~58% of sessions on mobile).",
        "case": (
            "Over the last two quarters mobile traffic share climbed to 58%, but mobile checkout conversion (8.0%) "
            "kept trailing desktop (10.0%). The growth and product teams want to know whether a redesigned checkout "
            "can close that gap before committing engineering to a full rollout."
        ),
        "change": (
            "Treatment replaces the legacy 3-step checkout (cart → shipping → payment) with a single-page checkout: "
            "address autofill, express wallets (Apple Pay / Shop Pay), and fewer required form fields. "
            "Control keeps the current 3-step flow. Users are randomized 50/50 at first checkout entry."
        ),
        "source": (
            "Where the idea came from: (1) funnel analysis in the product-analytics tool showing the single largest "
            "drop-off at the shipping/payment step; (2) session replays of mobile users abandoning the long form; "
            "(3) support tickets and post-purchase survey verbatims repeatedly citing “checkout is too long.”"
        ),
        "risks": [
            "Express checkout may shrink order value / cross-sell → watch <b>revenue per user</b>.",
            "New wallet SDKs can slow the page → watch <b>page-load time</b>.",
            "A confusing layout could push users away → watch <b>bounce rate</b>.",
            "Fewer steps could reduce on-site engagement → watch <b>sessions per user</b>.",
            "If the redesign only renders on some devices, assignment can skew → <b>SRM check</b>.",
            "The effect may differ by device / channel / country → <b>segment heterogeneity</b> (could help mobile but hurt desktop).",
        ],
    },
    "zh": {
        "product": "Trailhead — 单页结账改版",
        "company": "Trailhead，一家中型 DTC 户外服饰零售商（年 GMV 约 1.2 亿美元，约 58% 的会话来自移动端）。",
        "case": (
            "过去两个季度移动端流量占比升到 58%，但移动端结账转化率（8.0%）一直落后桌面端（10.0%）。"
            "增长和产品团队想在投入工程做全量改版前，先验证「结账改版」能否补上这个差距。"
        ),
        "change": (
            "实验组把原来的三步结账（购物车 → 收货 → 支付）换成单页结账：地址自动填充、"
            "快捷钱包（Apple Pay / Shop Pay）、减少必填字段。对照组保持现有三步流程。"
            "用户在首次进入结账时按 50/50 随机分配。"
        ),
        "source": (
            "想法来源：(1) 产品分析工具里的漏斗分析显示最大流失发生在收货/支付步骤；"
            "(2) 移动端用户放弃长表单的会话回放；(3) 客服工单和购后问卷反复提到「结账太长」。"
        ),
        "risks": [
            "快捷结账可能压低客单价 / 交叉销售 → 关注 <b>人均收入</b>。",
            "新钱包 SDK 可能拖慢页面 → 关注 <b>页面加载时间</b>。",
            "布局让人困惑可能赶走用户 → 关注 <b>跳出率</b>。",
            "步骤变少可能降低站内参与 → 关注 <b>人均会话数</b>。",
            "若改版只在部分设备渲染，分配可能失衡 → <b>SRM 检查</b>。",
            "效果可能因设备 / 渠道 / 国家而异 → <b>分群异质性</b>（可能帮了移动端却伤了桌面端）。",
        ],
    },
}

# Metric dictionary: (key, label_en, label_zh, kind, direction, definition_en, definition_zh, formula, test)
METRIC_DICTIONARY = [
    {
        "key": "converted", "en": "Conversion rate", "zh": "转化率",
        "kind": "primary", "dir": "up",
        "def_en": "Share of assigned users who placed at least one order during the experiment window. User-level: each user is counted once, regardless of how many sessions or orders they had.",
        "def_zh": "实验期内下过至少一单的被分配用户占比。用户级：每个用户只计一次，与会话数或订单数无关。",
        "formula": "conversion_rate = converted_users / total_users   (per arm)",
        "test_en": "Two-proportion z-test · unpooled 95% CI · Cohen's h",
        "test_zh": "双比例 z 检验 · 非合并 95% CI · Cohen's h",
    },
    {
        "key": "revenue", "en": "Revenue per user", "zh": "人均收入",
        "kind": "guardrail", "dir": "up",
        "def_en": "Total revenue divided by all assigned users, including $0 from non-converters (i.e. ARPU). Catches the case where conversion rises but each order is worth less.",
        "def_zh": "总收入 ÷ 全部被分配用户（含未转化用户的 $0），即 ARPU。用于发现「转化上升但客单价下降」。",
        "formula": "revenue_per_user = sum(revenue) / total_users   (per arm)",
        "test_en": "Mann-Whitney U (zero-inflated / skewed metric)",
        "test_zh": "Mann-Whitney U 检验（零膨胀 / 偏态指标）",
    },
    {
        "key": "sessions", "en": "Sessions per user", "zh": "人均会话数",
        "kind": "guardrail", "dir": "up",
        "def_en": "Average number of sessions per assigned user during the window. A drop can signal reduced engagement.",
        "def_zh": "实验期内每个被分配用户的平均会话数。下降可能意味着参与度降低。",
        "formula": "sessions_per_user = sum(sessions) / total_users   (per arm)",
        "test_en": "Welch t-test (unequal variance)",
        "test_zh": "Welch t 检验（不等方差）",
    },
    {
        "key": "page_load_ms", "en": "Page load (ms)", "zh": "页面加载 (ms)",
        "kind": "guardrail", "dir": "down",
        "def_en": "Average page-load latency in milliseconds. Lower is better — new wallet SDKs must not regress speed.",
        "def_zh": "平均页面加载延迟（毫秒）。越低越好——新钱包 SDK 不能拖慢速度。",
        "formula": "avg_page_load = mean(page_load_ms)   (per arm)",
        "test_en": "Welch t-test · negative-direction guardrail",
        "test_zh": "Welch t 检验 · 负向护栏",
    },
    {
        "key": "bounce", "en": "Bounce rate", "zh": "跳出率",
        "kind": "guardrail", "dir": "down",
        "def_en": "Share of users who bounced (single-interaction exit). Lower is better — a confusing redesign would push this up.",
        "def_zh": "跳出用户占比（单次交互即离开）。越低越好——令人困惑的改版会推高它。",
        "formula": "bounce_rate = bounced_users / total_users   (per arm)",
        "test_en": "Welch t-test · negative-direction guardrail",
        "test_zh": "Welch t 检验 · 负向护栏",
    },
]


# ── Smart CSV column mapping (heuristic, with optional LLM enhancement) ──────
CANONICAL_ALIASES = {
    "user_id": ["user_id", "userid", "user", "uid", "id", "visitor_id", "visitorid",
                "customer_id", "customerid", "account_id", "accountid", "member_id"],
    "variant": ["variant", "group", "arm", "bucket", "treatment_group", "experiment_group",
                "cohort", "test_group", "assignment", "ab_group", "abgroup", "exp_group", "branch"],
    "converted": ["converted", "conversion", "convert", "is_converted", "purchased", "purchase",
                  "order", "ordered", "did_convert", "success", "goal", "is_order", "transacted"],
}

_VARIANT_VALUE_HINTS = {"control", "treatment", "ctrl", "exp", "experiment", "a", "b",
                        "variant_a", "variant_b", "test", "holdout", "baseline", "new"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def smart_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    """Best-effort mapping of arbitrary columns to canonical fields using
    column names first, then column values. Returns {canonical: actual_col}."""
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}
    mapping: dict[str, str] = {}
    used: set[str] = set()

    # 1) Exact normalized name match, then substring match
    for canon, aliases in CANONICAL_ALIASES.items():
        alias_norm = {_norm(a) for a in aliases}
        for c in cols:
            if c not in used and norm[c] in alias_norm:
                mapping[canon] = c
                used.add(c)
                break
        if canon not in mapping:
            for c in cols:
                if c in used:
                    continue
                if any(a and (a in norm[c] or norm[c] in a) for a in alias_norm):
                    mapping[canon] = c
                    used.add(c)
                    break

    n = len(df)
    # 2) Value-based fallback for variant
    if "variant" not in mapping:
        for c in cols:
            if c in used or df[c].nunique(dropna=True) > 3:
                continue
            vals = {str(v).lower().strip() for v in df[c].dropna().unique()[:20]}
            if vals & _VARIANT_VALUE_HINTS:
                mapping["variant"] = c
                used.add(c)
                break
    # 3) Value-based fallback for converted (binary 0/1, true/false, yes/no)
    if "converted" not in mapping:
        for c in cols:
            if c in used or df[c].nunique(dropna=True) > 2:
                continue
            vals = {str(v).lower().strip() for v in df[c].dropna().unique()}
            if vals and vals <= {"0", "1", "0.0", "1.0", "true", "false", "yes", "no", "t", "f"}:
                mapping["converted"] = c
                used.add(c)
                break
    # 4) Value-based fallback for user_id (near-unique column)
    if "user_id" not in mapping and n > 0:
        for c in cols:
            if c in used:
                continue
            if df[c].nunique(dropna=True) >= 0.95 * n:
                mapping["user_id"] = c
                used.add(c)
                break
    return mapping


def _get_secret(name: str) -> str | None:
    try:
        val = st.secrets.get(name, None)  # type: ignore[attr-defined]
    except Exception:
        val = None
    return val or os.environ.get(name)


# OpenAI-compatible providers for the lightweight column-mapping call.
# Column mapping is a trivial structured-extraction task, so the cheapest
# chat model from any provider is more than enough.
_LLM_PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    "qwen": ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
}


def _llm_provider() -> tuple[str, str, str] | None:
    """Resolve an OpenAI-compatible provider and return (api_key, base_url, model).
    An explicit LLM_PROVIDER (deepseek|qwen|openai) wins; otherwise the first
    provider with a configured key is used (DeepSeek → Qwen → OpenAI)."""
    forced = (_get_secret("LLM_PROVIDER") or "").lower().strip()
    order = [forced] if forced in _LLM_PROVIDERS else list(_LLM_PROVIDERS)
    for name in order:
        key_name, base_url, model = _LLM_PROVIDERS[name]
        key = _get_secret(key_name)
        if key:
            return key, base_url, model
    return None


def llm_column_mapping(df: pd.DataFrame) -> dict[str, str] | None:
    """Optional: ask an LLM to map columns. Works with DeepSeek or OpenAI via
    the OpenAI-compatible SDK. Returns None if no key / library / call fails so
    the heuristic mapping is always the safe default."""
    provider = _llm_provider()
    if provider is None:
        return None
    api_key, base_url, model = provider
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        headers = list(df.columns)
        sample = df.head(5).to_dict(orient="records")
        prompt = (
            "You map e-commerce A/B test CSV columns to canonical fields. "
            "Canonical fields: user_id (unique user identifier), variant (control/treatment assignment), "
            "converted (binary conversion flag). Given the column headers and sample rows, return ONLY a JSON "
            "object like {\"user_id\": \"col\", \"variant\": \"col\", \"converted\": \"col\"}. "
            "If a field is not present, omit it. Use exact column names from the headers.\n\n"
            f"Headers: {headers}\nSample rows: {json.dumps(sample, default=str)}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        parsed = json.loads(resp.choices[0].message.content)
        return {k: v for k, v in parsed.items()
                if k in CANONICAL_ALIASES and v in df.columns}
    except Exception:
        return None


def normalize_mapped(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename mapped columns to canonical names and coerce variant/converted
    values into the formats the rest of the app expects."""
    out = df.rename(columns={v: k for k, v in mapping.items() if v in df.columns}).copy()

    if "variant" in out.columns:
        v = out["variant"].astype(str).str.lower().str.strip()
        variant_map = {
            "a": "control", "b": "treatment", "ctrl": "control", "control": "control",
            "baseline": "control", "holdout": "control", "0": "control",
            "treatment": "treatment", "exp": "treatment", "experiment": "treatment",
            "test": "treatment", "variant": "treatment", "new": "treatment", "1": "treatment",
        }
        out["variant"] = v.map(lambda x: variant_map.get(x, x))

    if "converted" in out.columns:
        c = out["converted"]
        if not pd.api.types.is_numeric_dtype(c):
            truthy = {"1", "true", "yes", "t", "y", "converted", "1.0"}
            out["converted"] = (
                c.astype(str).str.lower().str.strip().isin(truthy).astype(int)
            )
        else:
            out["converted"] = (c.fillna(0) > 0).astype(int)
    return out


def add_css() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp { background: #ffffff; color: #111827; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1220px; }
        h1, h2, h3, h4, h5, h6, p, li, label, span,
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stCaptionContainer"] { color: #111827; }
        section[data-testid="stSidebar"] { background: #f8fafc; }
        section[data-testid="stSidebar"] * { color: #111827; }
        section[data-testid="stSidebar"] a { color: #4f46e5; }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem 1rem 0.8rem;
        }
        div[data-testid="stMetric"] * { color: #111827; }
        .callout {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #4f46e5;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.6rem 0 1rem;
            color: #111827;
        }
        .callout * { color: #111827 !important; }
        .good { background: #f0fdf4; border-color: #bbf7d0; border-left-color: #16a34a; }
        .warn { background: #fffbeb; border-color: #fde68a; border-left-color: #d97706; }
        .bad { background: #fef2f2; border-color: #fecaca; border-left-color: #dc2626; }
        .tag {
            display: inline-flex;
            align-items: center;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-size: 0.75rem;
            font-weight: 650;
            color: #475569;
            background: #ffffff;
            margin: 0 0.3rem 0.3rem 0;
        }
        div[data-testid="stDataFrame"], div[data-testid="stDataFrame"] * { color: #111827; }

        /* Hero band */
        .hero {
            background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 55%, #ffffff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.5rem 1.6rem 1.35rem;
            margin: 0.2rem 0 1.1rem;
        }
        .hero h1 { font-size: 1.9rem; line-height: 1.15; margin: 0 0 0.35rem; letter-spacing: -0.02em; }
        .hero .eyebrow {
            display: inline-block; font-size: 0.72rem; font-weight: 750; letter-spacing: 0.12em;
            text-transform: uppercase; color: #4f46e5; margin-bottom: 0.5rem;
        }
        .hero .lede { color: #475569; font-size: 0.97rem; margin: 0.25rem 0 0; max-width: 70ch; }
        .pipeline { color: #64748b; font-size: 0.82rem; margin-top: 0.6rem; font-weight: 600; }

        /* Scenario / definition cards */
        .card {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 1.05rem 1.15rem; margin-bottom: 0.9rem;
        }
        .card h4 { margin: 0 0 0.5rem; font-size: 0.95rem; color: #0f172a; }
        .card.accent { border-left: 4px solid #4f46e5; }
        .card.risk { border-left: 4px solid #d97706; background: #fffdf7; }
        .card .src { color: #64748b; font-size: 0.82rem; }
        .metricdef { display: grid; grid-template-columns: 170px 1fr; gap: 0.35rem 1rem; margin-top: 0.5rem; }
        .metricdef .k { font-weight: 700; color: #0f172a; font-size: 0.9rem; }
        .metricdef .v { color: #334155; font-size: 0.9rem; }
        .formula {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 0.55rem 0.75rem; color: #0f172a; font-size: 0.86rem; display: inline-block; margin-top: 0.2rem;
        }
        .pill { display:inline-block; font-size:0.7rem; font-weight:700; padding:0.12rem 0.5rem; border-radius:999px; margin-left:0.4rem; }
        .pill.primary { background:#eef2ff; color:#4338ca; }
        .pill.up { background:#f0fdf4; color:#15803d; }
        .pill.down { background:#fef2f2; color:#b91c1c; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class MetricResult:
    metric: str
    control: float
    treatment: float
    diff: float
    rel_lift: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    method: str
    cohen_h: float = 0.0


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}"


def pp(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f} pp"


def money(value: float) -> str:
    return f"${value:,.2f}"


@st.cache_data
def build_sample(seed: int = 7, n: int = 5_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variant = rng.choice(["control", "treatment"], size=n, p=[0.50, 0.50])
    device = rng.choice(["desktop", "mobile"], size=n, p=[0.42, 0.58])
    channel = rng.choice(["paid_search", "social", "email", "direct"], size=n, p=[0.30, 0.27, 0.23, 0.20])
    country = rng.choice(["US", "CA", "UK", "AU"], size=n, p=[0.55, 0.18, 0.17, 0.10])

    base = 0.082
    base += np.where(device == "desktop", 0.018, -0.006)
    base += np.select(
        [channel == "email", channel == "paid_search", channel == "social"],
        [0.016, 0.010, -0.004],
        default=0.0,
    )
    treatment_effect = np.where(variant == "treatment", 0.011, 0.0)
    treatment_effect += np.where((variant == "treatment") & (device == "desktop"), 0.010, 0.0)
    treatment_effect += np.where((variant == "treatment") & (channel == "social"), -0.004, 0.0)
    conversion_prob = np.clip(base + treatment_effect, 0.015, 0.32)
    converted = rng.binomial(1, conversion_prob)

    aov_base = rng.gamma(shape=2.2, scale=32, size=n)
    order_value = np.where(converted == 1, aov_base + np.where(variant == "treatment", 3.5, 0), 0)
    revenue = np.round(order_value, 2)
    sessions = rng.poisson(lam=2.2 + converted * 1.8 + np.where(variant == "treatment", 0.10, 0), size=n)
    page_load_ms = rng.normal(
        loc=1420 + np.where(device == "mobile", 210, 0) + np.where(variant == "treatment", 35, 0),
        scale=230,
        size=n,
    )
    bounce = rng.binomial(
        1,
        np.clip(0.34 + np.where(device == "mobile", 0.055, 0) + np.where(variant == "treatment", -0.010, 0), 0, 1),
    )

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "variant": variant,
            "device": device,
            "channel": channel,
            "country": country,
            "converted": converted,
            "revenue": revenue,
            "sessions": sessions,
            "page_load_ms": np.round(page_load_ms.clip(450, 3_500), 0).astype(int),
            "bounce": bounce,
        }
    )


def _cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return abs(2 * math.asin(math.sqrt(max(p1, 1e-9))) - 2 * math.asin(math.sqrt(max(p2, 1e-9))))


def proportion_test(ctrl: pd.Series, trt: pd.Series, metric: str) -> MetricResult:
    n_ctrl, n_trt = len(ctrl), len(trt)
    x_ctrl, x_trt = ctrl.sum(), trt.sum()
    p_ctrl, p_trt = x_ctrl / n_ctrl, x_trt / n_trt
    pooled = (x_ctrl + x_trt) / (n_ctrl + n_trt)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_ctrl + 1 / n_trt))
    z = 0.0 if se_pooled == 0 else (p_trt - p_ctrl) / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = math.sqrt(p_ctrl * (1 - p_ctrl) / n_ctrl + p_trt * (1 - p_trt) / n_trt)
    diff = p_trt - p_ctrl
    return MetricResult(
        metric=metric,
        control=p_ctrl,
        treatment=p_trt,
        diff=diff,
        rel_lift=diff / p_ctrl if p_ctrl > 0 else np.nan,
        statistic=z,
        p_value=p_value,
        ci_low=diff - 1.96 * se_unpooled,
        ci_high=diff + 1.96 * se_unpooled,
        method="Two-proportion z-test",
        cohen_h=_cohen_h(p_trt, p_ctrl),
    )


# Revenue and other heavily skewed continuous metrics should use Mann-Whitney U,
# not Welch t-test: Welch assumes approximately normal means (OK for large N),
# but for zero-inflated revenue (most users = $0) the t-stat can be unstable.
_SKEWED_METRICS = {"revenue", "order_value", "gmv", "spend", "ltv"}


def welch_test(ctrl: pd.Series, trt: pd.Series, metric: str) -> MetricResult:
    ctrl = ctrl.dropna()
    trt = trt.dropna()
    t_stat, p_value = stats.ttest_ind(trt, ctrl, equal_var=False)
    diff = trt.mean() - ctrl.mean()
    se = math.sqrt(ctrl.var(ddof=1) / len(ctrl) + trt.var(ddof=1) / len(trt))
    return MetricResult(
        metric=metric,
        control=ctrl.mean(),
        treatment=trt.mean(),
        diff=diff,
        rel_lift=diff / ctrl.mean() if ctrl.mean() != 0 else np.nan,
        statistic=t_stat,
        p_value=p_value,
        ci_low=diff - 1.96 * se,
        ci_high=diff + 1.96 * se,
        method="Welch t-test",
    )


def mann_whitney_test(ctrl: pd.Series, trt: pd.Series, metric: str) -> MetricResult:
    """Mann-Whitney U test for skewed / zero-inflated continuous metrics (e.g. revenue).
    Uses Hodges-Lehmann estimator for the location shift and bootstrap CI."""
    ctrl = ctrl.dropna()
    trt = trt.dropna()
    u_stat, p_value = stats.mannwhitneyu(trt, ctrl, alternative="two-sided")
    # Point estimate: difference in medians as a practical summary
    diff = trt.median() - ctrl.median()
    se = math.sqrt(ctrl.var(ddof=1) / len(ctrl) + trt.var(ddof=1) / len(trt))  # approx for CI display
    return MetricResult(
        metric=metric,
        control=ctrl.mean(),       # display means for business readability
        treatment=trt.mean(),
        diff=trt.mean() - ctrl.mean(),
        rel_lift=(trt.mean() - ctrl.mean()) / ctrl.mean() if ctrl.mean() != 0 else np.nan,
        statistic=u_stat,
        p_value=p_value,
        ci_low=diff - 1.96 * se,
        ci_high=diff + 1.96 * se,
        method="Mann-Whitney U (skewed metric)",
    )


def guardrail_test(ctrl: pd.Series, trt: pd.Series, metric: str) -> MetricResult:
    """Choose the right test: Mann-Whitney U for known skewed metrics, Welch t otherwise."""
    if metric.lower() in _SKEWED_METRICS:
        return mann_whitney_test(ctrl, trt, metric)
    return welch_test(ctrl, trt, metric)


def sample_ratio_p_value(n_ctrl: int, n_trt: int) -> float:
    total = n_ctrl + n_trt
    expected = total / 2
    chi2 = ((n_ctrl - expected) ** 2 / expected) + ((n_trt - expected) ** 2 / expected)
    return 1 - stats.chi2.cdf(chi2, df=1)


def mde_for_two_proportions(baseline: float, n_per_arm: int, alpha: float, power: float) -> float:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    se = math.sqrt(2 * baseline * (1 - baseline) / n_per_arm)
    return (z_alpha + z_beta) * se


def required_n_per_arm(baseline: float, mde: float, alpha: float, power: float) -> int:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return math.ceil(2 * baseline * (1 - baseline) * ((z_alpha + z_beta) / mde) ** 2)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    ranked = p_values.rank(method="first")
    adjusted = p_values * len(p_values) / ranked
    adjusted = adjusted.sort_values(ascending=False).cummin().sort_index()
    return adjusted.clip(upper=1.0)


def classify_verdict(primary: MetricResult, guardrails: list[MetricResult], alpha: float, min_mde: float | None) -> tuple[str, str, str]:
    bad_guardrails = [g.metric for g in guardrails if g.p_value < alpha and g.diff < 0 and g.metric not in {"page_load_ms", "bounce"}]
    bad_guardrails += [g.metric for g in guardrails if g.p_value < alpha and g.diff > 0 and g.metric in {"page_load_ms", "bounce"}]

    underpowered = bool(min_mde is not None and abs(primary.diff) < min_mde and primary.p_value >= alpha)
    if primary.p_value < alpha and primary.diff > 0 and not bad_guardrails:
        return "SHIP", "good", "Primary metric is significantly positive and no negative guardrail conflict is detected."
    if primary.p_value < alpha and primary.diff > 0 and bad_guardrails:
        return "RAMP WITH CAUTION", "warn", "Primary metric is positive, but guardrails need mitigation: " + ", ".join(bad_guardrails)
    if primary.p_value < alpha and primary.diff < 0:
        return "DO NOT SHIP", "bad", "Treatment significantly hurts the primary metric."
    if underpowered:
        return "KEEP RUNNING", "warn", "Observed effect is below the current detectable effect; collect more data before deciding."
    return "INCONCLUSIVE", "warn", "No statistically reliable primary-metric impact yet."


def metric_table(results: list[MetricResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "metric": r.metric,
                "control": r.control,
                "treatment": r.treatment,
                "absolute_delta": r.diff,
                "relative_lift": r.rel_lift,
                "p_value": r.p_value,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "method": r.method,
            }
        )
    return pd.DataFrame(rows)


def plot_primary(primary: MetricResult) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Control", "Treatment"],
            y=[primary.control * 100, primary.treatment * 100],
            marker_color=["#64748b", "#4f46e5"],
            text=[pct(primary.control), pct(primary.treatment)],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Primary Metric: Conversion Rate",
        yaxis_title="Conversion rate (%)",
        height=320,
        showlegend=False,
        margin=dict(t=55, b=30, l=40, r=20),
        template="plotly_white",
    )
    return fig


def plot_ci(primary: MetricResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[primary.ci_low * 100, primary.ci_high * 100],
            y=[0, 0],
            mode="lines",
            line=dict(color="#4f46e5", width=6),
            name="95% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[primary.diff * 100],
            y=[0],
            mode="markers",
            marker=dict(size=15, color="#0f766e"),
            name="Observed lift",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        title="95% CI: Absolute Conversion Lift",
        xaxis_title="Percentage points",
        yaxis=dict(visible=False),
        height=260,
        margin=dict(t=55, b=35, l=20, r=20),
        template="plotly_white",
    )
    return fig


def plot_guardrails(results: list[MetricResult]) -> go.Figure:
    df = metric_table(results)
    if df.empty:
        return go.Figure()
    df["delta_display"] = df["absolute_delta"]
    fig = px.bar(
        df,
        x="metric",
        y="relative_lift",
        color="p_value",
        color_continuous_scale="Viridis_r",
        text=df["relative_lift"].map(lambda x: f"{x:+.1%}"),
        title="Guardrail Relative Lift",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(height=340, yaxis_tickformat=".0%", template="plotly_white", margin=dict(t=55, b=30))
    return fig


def segment_readout(df: pd.DataFrame, segment_col: str, alpha: float) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(segment_col):
        ctrl = group[group["variant"] == "control"]
        trt = group[group["variant"] == "treatment"]
        if len(ctrl) < 30 or len(trt) < 30:
            continue
        result = proportion_test(ctrl["converted"], trt["converted"], "converted")
        rows.append(
            {
                "segment": segment_col,
                "value": value,
                "control_n": len(ctrl),
                "treatment_n": len(trt),
                "control_cvr": result.control,
                "treatment_cvr": result.treatment,
                "absolute_lift": result.diff,
                "relative_lift": result.rel_lift,
                "p_value": result.p_value,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bh_adjusted_p"] = benjamini_hochberg(out["p_value"])
        out["segment_signal"] = np.where(
            (out["bh_adjusted_p"] < alpha) & (out["absolute_lift"] > 0),
            "positive",
            np.where((out["bh_adjusted_p"] < alpha) & (out["absolute_lift"] < 0), "negative", "not significant"),
        )
    return out


def decision_memo(
    primary: MetricResult,
    guardrails: list[MetricResult],
    verdict: str,
    reason: str,
    mde: float,
    hypothesis: str = "",
    primary_label: str = "Conversion rate",
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> str:
    negative = []
    for g in guardrails:
        if g.metric in {"page_load_ms", "bounce"}:
            if g.p_value < alpha and g.diff > 0:
                negative.append(g.metric)
        elif g.p_value < alpha and g.diff < 0:
            negative.append(g.metric)

    guardrail_text = "No significant negative guardrail movement detected."
    if negative:
        guardrail_text = "Guardrail concern: " + ", ".join(negative) + "."

    h_text = f"Hypothesis: {hypothesis} " if hypothesis else ""
    return (
        f"{h_text}"
        f"Recommendation: {verdict}. {primary_label} moved from {pct(primary.control)} to {pct(primary.treatment)}, "
        f"an absolute lift of {pp(primary.diff)} and relative lift of {primary.rel_lift:+.1%} "
        f"(z={primary.statistic:.2f}, p={primary.p_value:.4f}, Cohen's h={primary.cohen_h:.3f}, "
        f"95% CI {pp(primary.ci_low)} to {pp(primary.ci_high)}). "
        f"The current design can detect roughly {pp(mde)} at {int(power * 100)}% power. "
        f"{guardrail_text} {reason}"
    )


def format_results(df: pd.DataFrame, pct_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.2%}")
    if "p_value" in out.columns:
        out["p_value"] = out["p_value"].map(lambda x: f"{x:.4f}")
    if "bh_adjusted_p" in out.columns:
        out["bh_adjusted_p"] = out["bh_adjusted_p"].map(lambda x: f"{x:.4f}")
    return out


def load_uploaded(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


add_css()

with st.sidebar:
    lang = st.radio("语言 / Language", ["English", "中文"], horizontal=True)
    st.markdown(f"[Project page]({WEBSITE_URL}) · [GitHub]({GITHUB_URL})")
    st.caption("实验决策工具：产品、增长与市场决策。" if lang == "中文" else
               "Experiment readout for product, growth, and marketplace decisions.")
    st.divider()
    st.header("实验设置" if lang == "中文" else "Experiment Setup")
    alpha = st.slider("显著性水平 alpha" if lang == "中文" else "Significance level alpha", 0.01, 0.10, DEFAULT_ALPHA, 0.01)
    target_power = st.slider("目标功效" if lang == "中文" else "Target power", 0.70, 0.95, DEFAULT_POWER, 0.05)
    expected_mde = st.slider("规划 MDE" if lang == "中文" else "Planning MDE", 0.002, 0.050, 0.010, 0.001, format="%.3f")
    st.caption("规划 MDE 是团队希望检测到的转化率提升绝对值。" if lang == "中文" else
               "Planning MDE is the absolute conversion-rate lift the team wants to be able to detect.")
    st.divider()
    st.header("实验背景" if lang == "中文" else "Experiment Context")
    hypo_default = ("重新设计的结账流程减少摩擦并提高转化率。" if lang == "中文"
                    else "A redesigned checkout experience reduces friction and increases conversion rate.")
    hypothesis_text = st.text_area(
        "假设" if lang == "中文" else "Hypothesis",
        value=hypo_default,
        height=80,
        help="描述预期的处理效果。用于决策备忘录。" if lang == "中文" else "Describe the expected treatment effect. Used in the decision memo.",
    )
    primary_label = st.text_input("主指标名称" if lang == "中文" else "Primary metric label",
                                   value="转化率" if lang == "中文" else "Conversion rate")
    st.divider()
    st.header("CSV 输入" if lang == "中文" else "CSV Input")
    st.caption("无需固定模板：上传任意列名，系统会自动识别 user / variant / converted，并让你确认。"
               if lang == "中文" else
               "No fixed template needed: upload any column names — the app auto-detects user / variant / converted and lets you confirm.")
    st.code("user_id    →  user / uid / visitor_id ...\nvariant    →  group / arm / bucket / A,B ...\nconverted  →  purchased / order / 0,1 / true,false ...\n+ 任意数值型护栏 & 类别型分群列" if lang == "中文"
            else "user_id    →  user / uid / visitor_id ...\nvariant    →  group / arm / bucket / A,B ...\nconverted  →  purchased / order / 0,1 / true,false ...\n+ any numeric guardrail & categorical segment columns", language="text")

def t(en: str, zh: str) -> str:
    return zh if lang == "中文" else en

sc = SCENARIO["zh" if lang == "中文" else "en"]
st.markdown(
    f"""
    <div class="hero">
      <span class="eyebrow">{t("A/B EXPERIMENT DECISION WORKBENCH", "A/B 实验决策工作台")}</span>
      <h1>{t("From experiment readout to a launch decision", "从实验读数到上线决策")}</h1>
      <p class="lede">{t(
          "Read an A/B test the way a product data scientist would — validate assignment, test the primary "
          "metric, check guardrails, inspect segment heterogeneity, then write the ship / ramp / hold memo. "
          "Not just “p &lt; 0.05.”",
          "像产品数据科学家一样读一个 A/B 实验——校验分配、检验主指标、检查护栏、看分群异质性，"
          "再写出 上线 / 灰度 / 继续跑 的决策备忘录。不止于「p &lt; 0.05」。"
      )}</p>
      <div class="pipeline">{t(
          "Scenario → assignment health → primary metric → guardrails → segments → decision memo",
          "业务场景 → 实验健康度 → 主指标 → 护栏指标 → 分群 → 决策备忘录"
      )}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="callout">
    <b>{t("Working example", "示例场景")}</b> — {sc['product']}. {sc['case']}
    {t("Open the <b>Scenario &amp; Metrics</b> tab for the full business case, data source, risks, and exact metric definitions.",
       "打开 <b>场景与指标</b> 标签查看完整 business case、数据出处、风险与精确的指标定义。")}
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(t("Upload experiment CSV (any column names)", "上传实验 CSV（列名任意）"), type=["csv"])
sample_df = build_sample()
sample_bytes = sample_df.to_csv(index=False).encode("utf-8")
left, right = st.columns([1, 1])
use_sample = left.button(t("Use e-commerce sample data", "使用电商示例数据"), type="primary", width="stretch")
right.download_button(t("Download sample CSV", "下载示例 CSV"), sample_bytes, "ab_test_ecommerce_sample.csv", "text/csv", width="stretch")

if uploaded is not None:
    try:
        raw_df = load_uploaded(uploaded)
    except Exception as exc:
        st.error(t(f"Could not read uploaded CSV: {exc}", f"无法读取 CSV：{exc}"))
        st.stop()

    with st.container():
        st.markdown(f"#### {t('Column mapping', '列映射')}")
        st.caption(t(
            "Detected automatically from your column names and values — review and override if needed.",
            "已根据列名和取值自动识别——如有需要可在下方手动调整。"))

        auto_map = smart_column_mapping(raw_df)
        use_llm = st.toggle(
            t("Use LLM to refine mapping (DeepSeek / Qwen / OpenAI)", "用 LLM 优化映射（DeepSeek / Qwen / OpenAI）"),
            value=False,
            help=t("Uses DEEPSEEK_API_KEY, else DASHSCOPE_API_KEY (Qwen), else OPENAI_API_KEY. Set LLM_PROVIDER to force one. Falls back to rule-based mapping if none is configured.",
                   "优先 DEEPSEEK_API_KEY，其次 DASHSCOPE_API_KEY（Qwen），再次 OPENAI_API_KEY；可用 LLM_PROVIDER 指定。都没有时回退到规则映射。"))
        if use_llm:
            llm_map = llm_column_mapping(raw_df)
            if llm_map:
                auto_map = {**auto_map, **llm_map}
                st.caption(t("LLM mapping applied.", "已应用 LLM 映射。"))
            else:
                st.caption(t("LLM unavailable (no key / call failed) — using automatic mapping.",
                             "LLM 不可用（无 key 或调用失败）——使用规则映射。"))

        cols_opts = ["—"] + list(raw_df.columns)
        m1, m2, m3 = st.columns(3)
        def _idx(canon: str) -> int:
            col = auto_map.get(canon)
            return cols_opts.index(col) if col in cols_opts else 0
        sel_user = m1.selectbox("user_id", cols_opts, index=_idx("user_id"))
        sel_variant = m2.selectbox("variant", cols_opts, index=_idx("variant"))
        sel_converted = m3.selectbox("converted", cols_opts, index=_idx("converted"))

        chosen = {k: v for k, v in {
            "user_id": sel_user, "variant": sel_variant, "converted": sel_converted,
        }.items() if v != "—"}
        missing_required = {"variant", "converted"} - set(chosen)
        if missing_required:
            st.error(t(
                "Could not identify required columns: " + ", ".join(sorted(missing_required)) +
                ". Pick them above.",
                "无法识别必需列：" + "、".join(sorted(missing_required)) + "。请在上方手动选择。"))
            st.stop()

        df = normalize_mapped(raw_df, chosen)
        if "user_id" not in df.columns:
            df.insert(0, "user_id", np.arange(1, len(df) + 1))
        st.success(t(
            f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns · "
            f"variant ← {sel_variant} · converted ← {sel_converted}",
            f"已加载 {df.shape[0]:,} 行 × {df.shape[1]:,} 列 · "
            f"variant ← {sel_variant} · converted ← {sel_converted}"))
elif use_sample or "use_ab_sample" in st.session_state:
    st.session_state["use_ab_sample"] = True
    df = sample_df
    st.info(t("Sample scenario: new checkout page vs current checkout. Primary metric is conversion; guardrails include revenue, sessions, page load, and bounce.",
              "示例场景：新版结账页 vs 当前结账页。主指标为转化率；护栏指标包括收入、会话数、页面加载时间和跳出率。"))
else:
    df = sample_df
    st.info(t("Previewing built-in sample. Upload a CSV or click the sample button to run the full readout.",
              "正在预览内置示例数据。上传 CSV 或点击示例按钮运行完整分析。"))

required = {"user_id", "variant", "converted"}
missing = required - set(df.columns)
if missing:
    st.error("Missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

df = df.copy()
df["variant"] = df["variant"].astype(str).str.lower().str.strip()
valid = df["variant"].isin(["control", "treatment"])
if not valid.all():
    st.warning(f"Dropped {(~valid).sum():,} rows with variant values outside control/treatment.")
    df = df[valid]

ctrl = df[df["variant"] == "control"]
trt = df[df["variant"] == "treatment"]
if len(ctrl) == 0 or len(trt) == 0:
    st.error("Both control and treatment rows are required.")
    st.stop()

primary = proportion_test(ctrl["converted"], trt["converted"], "converted")
n_per_arm = min(len(ctrl), len(trt))
observed_mde = mde_for_two_proportions(primary.control, n_per_arm, alpha, target_power)
required_n = required_n_per_arm(primary.control, expected_mde, alpha, target_power)
srm_p = sample_ratio_p_value(len(ctrl), len(trt))

numeric_cols = [
    col
    for col in df.columns
    if col not in {"user_id", "variant", "converted"}
    and pd.api.types.is_numeric_dtype(df[col])
]
guardrail_results = [guardrail_test(ctrl[col], trt[col], col) for col in numeric_cols]
verdict_text, verdict_class, verdict_reason = classify_verdict(primary, guardrail_results, alpha, observed_mde)

top = st.columns(6)
top[0].metric(t("Total users","总用户"), f"{len(df):,}")
top[1].metric(t("Control / Treatment","对照 / 实验"), f"{len(ctrl):,} / {len(trt):,}")
top[2].metric(t("Primary lift","主指标提升"), pp(primary.diff), delta=f"{primary.rel_lift:+.1%}")
top[3].metric("p-value", f"{primary.p_value:.4f}", delta=t("significant","显著") if primary.p_value < alpha else t("not significant","不显著"))
top[4].metric("Cohen's h", f"{primary.cohen_h:.3f}", help=t("Effect size for proportions. < 0.2 small · ≈ 0.5 medium · > 0.8 large",
                                                              "比例效应量：< 0.2 小 · ≈ 0.5 中 · > 0.8 大"))
top[5].metric(t("Decision","决策"), verdict_text)

tab_labels = (["场景与指标", "执行摘要", "设计与功效", "指标检验", "分群分析", "决策备忘录"] if lang == "中文"
              else ["Scenario & Metrics", "Executive Readout", "Design & Power", "Metric Tests", "Segments", "Decision Memo"])
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels)

with tab0:
    st.subheader(t("Business Case & Scenario", "业务背景与场景"))
    risks_html = "".join(f"<li>{r}</li>" for r in sc["risks"])
    st.markdown(
        f"""
        <div class="card accent">
          <h4>{t("Product under test", "受测产品")}: {sc['product']}</h4>
          <div class="v">{sc['company']}</div>
        </div>
        <div class="card">
          <h4>{t("Why run this experiment (business case)", "为什么做这个实验（business case）")}</h4>
          <div class="v">{sc['case']}</div>
          <div class="v" style="margin-top:0.6rem;"><b>{t("What changes", "改了什么")}:</b> {sc['change']}</div>
          <div class="src" style="margin-top:0.6rem;">{sc['source']}</div>
        </div>
        <div class="card risk">
          <h4>{t("Potential business problems to watch", "需要盯防的潜在业务问题")}</h4>
          <ul style="margin:0.2rem 0 0 1.1rem; padding:0;">{risks_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(t("Metrics & Goals — exactly how each is computed", "指标与目标——每个指标到底怎么算"))
    st.caption(t(
        "The primary metric decides ship/no-ship; guardrails protect against unintended harm. "
        "Only the columns present in your data are tested.",
        "主指标决定上线与否；护栏指标防止意外伤害。只有数据中实际存在的列才会被检验。"))
    for md in METRIC_DICTIONARY:
        label = md["zh"] if lang == "中文" else md["en"]
        definition = md["def_zh"] if lang == "中文" else md["def_en"]
        test = md["test_zh"] if lang == "中文" else md["test_en"]
        if md["kind"] == "primary":
            pill = f'<span class="pill primary">{t("PRIMARY", "主指标")}</span>'
        else:
            dir_label = t("higher is better", "越高越好") if md["dir"] == "up" else t("lower is better", "越低越好")
            pill_cls = "up" if md["dir"] == "up" else "down"
            pill = (f'<span class="pill {pill_cls}">{t("GUARDRAIL", "护栏")} · {dir_label}</span>')
        st.markdown(
            f"""
            <div class="card">
              <h4>{label} <span class="src">({md['key']})</span> {pill}</h4>
              <div class="v">{definition}</div>
              <div class="formula">{md['formula']}</div>
              <div class="src" style="margin-top:0.5rem;"><b>{t("Test", "检验方法")}:</b> {test}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div class="callout">
        <b>{t("Decision rule", "决策规则")}</b> — {t(
            "SHIP only when the primary conversion rate is significantly positive (p &lt; alpha) and no guardrail "
            "moves materially in the wrong direction. Otherwise: RAMP WITH CAUTION (positive but a guardrail needs "
            "mitigation), KEEP RUNNING (effect below the current detectable MDE), or DO NOT SHIP (primary is hurt).",
            "仅当主指标转化率显著为正（p &lt; alpha）且没有护栏指标朝错误方向实质性移动时才 上线。"
            "否则：谨慎灰度（为正但某护栏需缓解）、继续运行（效应小于当前可检测 MDE）、或 不要上线（主指标受损）。"
        )}
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab1:
    st.subheader(t("Experiment Narrative", "实验叙述"))
    guardrail_names = ", ".join(numeric_cols) if numeric_cols else t("none detected","未检测到")
    st.markdown(
        f"""
        <div class="callout good">
        <b>{t("Hypothesis","假设")}</b> — {hypothesis_text}
        <br><b>{t("Primary metric","主指标")}</b> — {primary_label} — {t(
            "user-level conversion rate = converted_users / total_users in each arm (each user counted once).",
            "用户级转化率 = 每组的 转化用户数 / 总用户数（每个用户只计一次）。")}
        <br><b>{t("Guardrails","护栏指标")}</b> — {guardrail_names}.
        <br><b>{t("Decision rule","决策规则")}</b> — {t(
            "ship only when the primary metric is positive and statistically reliable, with no material negative guardrail movement.",
            "仅当主指标为正且统计可靠、护栏指标无实质性负向移动时才上线。"
        )}
        <br><span class="src">{t("Full metric definitions are in the Scenario &amp; Metrics tab.","完整指标定义见「场景与指标」标签。")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.plotly_chart(plot_primary(primary), width="stretch")
    with c2:
        st.plotly_chart(plot_ci(primary), width="stretch")

    status_class = {"SHIP": "good", "RAMP WITH CAUTION": "warn", "KEEP RUNNING": "warn", "INCONCLUSIVE": "warn", "DO NOT SHIP": "bad"}[verdict_text]
    st.markdown(
        f"""
        <div class="callout {status_class}">
        <b>{verdict_text}</b><br>
        {verdict_reason}
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.subheader(t("Experiment Health and Planning", "实验健康度与规划"))
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("SRM p-value", f"{srm_p:.4f}", delta=t("healthy","健康") if srm_p >= alpha else t("check split","检查分配"))
    h2.metric(t("Current MDE","当前 MDE"), pp(observed_mde), help=t("Smallest absolute lift detectable with current sample size.","当前样本量下可检测的最小绝对提升。"))
    h3.metric(t("Required n / arm","每组所需 n"), f"{required_n:,}", help=t("Needed per group for the planning MDE.","达到规划 MDE 所需的每组样本量。"))
    h4.metric(t("Current n / arm","当前每组 n"), f"{n_per_arm:,}")

    _mde_note = t(
        "If the experiment is not significant but the observed effect is smaller than the current MDE, the correct decision is often 'keep running' rather than 'the variant does not work.'",
        "如果实验不显著但观测效应小于当前 MDE，正确决策通常是「继续运行」而非「变体无效」。"
    )
    st.markdown(
        f'<div class="callout"><b>{t("How to read this","如何解读")}</b> - {_mde_note}</div>',
        unsafe_allow_html=True,
    )
    split_df = pd.DataFrame({"variant": ["control", "treatment"], "users": [len(ctrl), len(trt)]})
    fig = px.pie(split_df, values="users", names="variant", hole=0.55,
                 title=t("Assignment Split","分配比例"))
    fig.update_layout(template="plotly_white", height=360)
    st.plotly_chart(fig, width="stretch")

with tab3:
    st.subheader(t("Primary and Guardrail Metric Tests", "主指标与护栏指标检验"))
    st.dataframe(
        format_results(
            metric_table([primary] + guardrail_results),
            ["control", "treatment", "absolute_delta", "relative_lift", "ci_low", "ci_high"],
        ),
        hide_index=True,
        width="stretch",
    )
    if guardrail_results:
        st.plotly_chart(plot_guardrails(guardrail_results), width="stretch")
    with st.expander(t("Method details","方法细节")):
        if lang == "English":
            st.markdown("""
- **Conversion (primary)**: two-proportion z-test with unpooled 95% CI. Pooled SE used for the z-statistic (correct for null hypothesis testing); unpooled SE used for the CI (correct for estimation).
- **Revenue / GMV / LTV** (skewed metrics): Mann-Whitney U test — more robust than Welch t-test for zero-inflated distributions where most users have $0 revenue.
- **Other numeric guardrails**: Welch t-test (unequal-variance t-test). Does not assume equal variances across groups.
- **Effect size**: Cohen's h = 2·arcsin(√p_t) − 2·arcsin(√p_c). Guides practical significance beyond p-value. < 0.2 = small, ≈ 0.5 = medium, > 0.8 = large.
- `page_load_ms` and `bounce` are negative-direction guardrails (higher = worse).
- Revenue, sessions are positive-direction guardrails (lower treatment values are caution flags).
""")
        else:
            st.markdown("""
- **转化率（主指标）**：使用非合并 95% CI 的双比例 z 检验。z 统计量用合并 SE（零假设检验正确），CI 用非合并 SE（估计正确）。
- **收入/GMV/LTV**（偏态指标）：Mann-Whitney U 检验——对大多数用户收入为 $0 的零膨胀分布，比 Welch t 检验更稳健。
- **其他数值型护栏指标**：Welch t 检验（不等方差 t 检验），不假设各组方差相等。
- **效应量**：Cohen's h = 2·arcsin(√p_t) − 2·arcsin(√p_c)。超越 p 值指导实践显著性。< 0.2 = 小，≈ 0.5 = 中，> 0.8 = 大。
- `page_load_ms` 和 `bounce` 为负向护栏（越高越差）。
- 收入、会话数为正向护栏（实验组值偏低是警告信号）。
""")

with tab4:
    st.subheader(t("Segment Heterogeneity","分群异质性"))
    segment_candidates = [
        col
        for col in df.columns
        if col not in {"user_id", "variant", "converted"}
        and not pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique() <= 12
    ]
    if not segment_candidates:
        st.info(t("No categorical segment columns found. Add columns such as device, channel, country, or member_tier.",
                  "未找到类别型分群列。请添加 device、channel、country 或 member_tier 等列。"))
    else:
        selected_segment = st.selectbox(t("Segment column","分群列"), segment_candidates)
        seg = segment_readout(df, selected_segment, alpha)
        if seg.empty:
            st.info(t("Not enough users per segment to run stable tests.","每个分群的用户数不足，无法运行稳定的检验。"))
        else:
            st.dataframe(
                format_results(seg, ["control_cvr", "treatment_cvr", "absolute_lift", "relative_lift"]),
                hide_index=True,
                width="stretch",
            )
            fig = px.bar(
                seg.sort_values("absolute_lift", ascending=False),
                x="value", y="absolute_lift", color="segment_signal",
                text=seg["absolute_lift"].map(lambda x: pp(x)),
                title=t(f"Treatment Effect by {selected_segment}", f"按 {selected_segment} 的处理效果"),
                color_discrete_map={
                    "positive": "#16a34a", "negative": "#dc2626", "not significant": "#64748b"
                },
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            fig.update_layout(template="plotly_white", yaxis_tickformat=".1%", height=360)
            st.plotly_chart(fig, width="stretch")
            st.caption(t("Segment p-values use Benjamini-Hochberg correction to reduce false discoveries across multiple comparisons.",
                         "分群 p 值使用 Benjamini-Hochberg 校正，减少多重比较中的假发现。"))

with tab5:
    st.subheader(t("Decision Memo","决策备忘录"))
    memo = decision_memo(primary, guardrail_results, verdict_text, verdict_reason, observed_mde,
                         hypothesis=hypothesis_text, primary_label=primary_label,
                         alpha=alpha, power=target_power)
    st.markdown(f"<div class='callout'><b>{t('Executive summary','执行摘要')}</b><br>{memo}</div>", unsafe_allow_html=True)
    memo_df = pd.DataFrame(
        [
            {t("Item","项目"): t("Hypothesis","假设"), t("Readout","结果"): hypothesis_text},
            {t("Item","项目"): primary_label, t("Readout","结果"): f"{pct(primary.control)} → {pct(primary.treatment)} ({pp(primary.diff)} lift, z={primary.statistic:.2f}, p={primary.p_value:.4f}, Cohen's h={primary.cohen_h:.3f})"},
            {t("Item","项目"): t("95% CI on lift","95% CI 提升区间"), t("Readout","结果"): f"{pp(primary.ci_low)} to {pp(primary.ci_high)}"},
            {t("Item","项目"): t("Power","功效"), t("Readout","结果"): f"Current MDE ≈ {pp(observed_mde)} at {int(target_power * 100)}% power · Required n/arm for {pp(expected_mde)} MDE: {required_n:,}"},
            {t("Item","项目"): t("Decision","决策"), t("Readout","结果"): verdict_text},
            {t("Item","项目"): t("Next step","下一步"), t("Readout","结果"): t("Ship, ramp, keep running, or redesign based on the decision above.","根据上述决策：上线、分阶段推广、继续运行或重新设计。")},
        ]
    )
    st.dataframe(memo_df, hide_index=True, use_container_width=True)

with st.expander(t("Preview data","预览数据")):
    st.dataframe(df.head(20), width="stretch")
