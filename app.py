"""
A/B Test Decision Workbench

Upload user-level experiment data or use the built-in e-commerce sample.
The app checks experiment design, primary metric significance, guardrails,
power/MDE, segment heterogeneity, and a ship/no-ship decision memo.
"""

from __future__ import annotations

import io
import math
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
        }
        div[data-testid="stDataFrame"], div[data-testid="stDataFrame"] * { color: #111827; }
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
    st.header("CSV 要求" if lang == "中文" else "CSV Requirements")
    st.code("user_id\nvariant: control/treatment\nconverted: 0/1\n可选数值型护栏指标" if lang == "中文"
            else "user_id\nvariant: control/treatment\nconverted: 0/1\noptional numeric guardrails", language="text")

def t(en: str, zh: str) -> str:
    return zh if lang == "中文" else en

st.title(t("A/B Test Decision Workbench", "A/B 测试决策工作台"))
st.caption(t("Hypothesis -> experiment health -> primary metric -> guardrails -> segments -> ship decision",
             "假设 → 实验健康度 → 主指标 → 护栏指标 → 分群 → 上线决策"))
st.markdown(
    f"""
    <div class="callout">
    <b>{t("Business question", "业务问题")}</b> - {t(
        "Should a new e-commerce checkout or landing-page experience ship? "
        "This workbench reads the experiment like a product data scientist: first validate assignment, "
        "then test the primary metric, check guardrails, review segment heterogeneity, and write the decision memo.",
        "新版结账/落地页是否应该上线？不是只看 p-value，而是把实验假设、主指标、"
        "护栏指标、样本量、分层效果和最终上线建议串成一个完整决策链。"
    )}
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(t("Upload experiment CSV", "上传实验 CSV"), type=["csv"])
sample_df = build_sample()
sample_bytes = sample_df.to_csv(index=False).encode("utf-8")
left, right = st.columns([1, 1])
use_sample = left.button(t("Use e-commerce sample data", "使用电商示例数据"), type="primary", width="stretch")
right.download_button(t("Download sample CSV", "下载示例 CSV"), sample_bytes, "ab_test_ecommerce_sample.csv", "text/csv", width="stretch")

if uploaded is not None:
    try:
        df = load_uploaded(uploaded)
        st.success(t(f"Loaded uploaded data: {df.shape[0]:,} rows x {df.shape[1]:,} columns",
                     f"已加载上传数据：{df.shape[0]:,} 行 × {df.shape[1]:,} 列"))
    except Exception as exc:
        st.error(t(f"Could not read uploaded CSV: {exc}", f"无法读取 CSV：{exc}"))
        st.stop()
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

tab_labels = (["执行摘要", "设计与功效", "指标检验", "分群分析", "决策备忘录"] if lang == "中文"
              else ["Executive Readout", "Design & Power", "Metric Tests", "Segments", "Decision Memo"])
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels)

with tab1:
    st.subheader(t("Experiment Narrative", "实验叙述"))
    guardrail_names = ", ".join(numeric_cols) if numeric_cols else t("none detected","未检测到")
    st.markdown(
        f"""
        <div class="callout good">
        <b>{t("Hypothesis","假设")}</b> — {hypothesis_text}
        <br><b>{t("Primary metric","主指标")}</b> — {primary_label} ({t("user-level conversion rate","用户级转化率")}).
        <br><b>{t("Guardrails","护栏指标")}</b> — {guardrail_names}.
        <br><b>{t("Decision rule","决策规则")}</b> — {t(
            "ship only when the primary metric is positive and statistically reliable, with no material negative guardrail movement.",
            "仅当主指标为正且统计可靠、护栏指标无实质性负向移动时才上线。"
        )}
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
