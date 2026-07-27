"""液压系统冷却器故障检测 · 置信带偏离检出 Demo"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="冷却器故障检测 Demo",
    page_icon="\U0001f6e0️",
    layout="wide",
)

# ============================================================
# Load data
# ============================================================
@st.cache_data
def load_data():
    with np.load("app_data/demo_windows.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

data = load_data()

# Extract metadata
tau_level = float(data["tau_level"])
tau_shape = float(data["tau_shape"])
a_det = float(data["a_detection"])
b_fpr = float(data["b_fpr"])
c_fpr = float(data["c_fpr"])

# Build dataframe for filtering
df = pd.DataFrame({
    "window_id": data["window_id"],
    "type": data["type"].astype(str),
    "X_valve": data["X_valve"],
    "X_pump": data["X_pump"],
    "X_accum": data["X_accum"],
    "cooler_target": data["cooler_target"],
    "is_calib": data["is_calib"],
    "s_level": data["s_level"],
    "s_shape": data["s_shape"],
    "verdict": data["verdict"].astype(str),
})
# Filter to test set only for demo
df = df[~df["is_calib"]].copy()

# ============================================================
# Title
# ============================================================
st.title("\U0001f6e0️ 液压系统冷却器故障检测 · 置信带偏离检出 Demo")
st.caption("基于 Chronos-Bolt 健康波形先验 + 确定性物理规则判决")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("窗口选择")

type_label_map = {
    "A": "故障注入 (A)",
    "B": "近距健康对照 (B)",
    "C": "远距健康对照 (C)",
}
type_choice = st.sidebar.selectbox(
    "窗口类型",
    options=["A", "B", "C"],
    format_func=lambda x: type_label_map[x],
)

# Filter by type
df_type = df[df["type"] == type_choice]

# X selection
df_type["X_label"] = df_type.apply(
    lambda r: f"valve={r['X_valve']}, pump_leak={r['X_pump']}, accum={r['X_accum']}bar", axis=1
)
x_options = sorted(df_type["X_label"].unique())
x_choice = st.sidebar.selectbox("工况配置 X (valve, pump_leak, accum)", options=x_options)

# Filter by X
df_x = df_type[df_type["X_label"] == x_choice]

# Cycle selection
n_cycles_avail = len(df_x)
cycle_choice = st.sidebar.slider(
    "cycle 序号", min_value=0, max_value=max(0, n_cycles_avail - 1),
    value=0, step=1,
)

st.sidebar.markdown("---")
st.sidebar.caption(f"该组合共 {n_cycles_avail} 个 cycle")

# ============================================================
# Get selected window data
# ============================================================
row = df_x.iloc[cycle_choice]
wid = int(row["window_id"])

ctx_tail = data["ctx_tail"][wid]
q10 = data["q10"][wid]
q50 = data["q50"][wid]
q90 = data["q90"][wid]
actual = data["actual"][wid]
s_level = float(row["s_level"])
s_shape = float(row["s_shape"])
verdict = row["verdict"]
cooler_tgt = int(row["cooler_target"])

# ============================================================
# Main area: Plot
# ============================================================
time_pred = np.arange(60)
time_ctx = np.arange(-60, 0)

# Compute residual
residual = np.abs(actual - q50)

# Find outside-band points
outside = (actual < q10) | (actual > q90)

# Build subplot with dual y-axis using secondary_y
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Left Y: temperature
fig.add_trace(
    go.Scatter(
        x=time_ctx, y=ctx_tail, mode="lines", name="Context (尾部 60s)",
        line=dict(color="gray", width=1.8),
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=time_pred, y=q10, mode="lines", name="q10",
        line=dict(color="#4575b4", width=0.5), showlegend=False,
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=time_pred, y=q90, mode="lines", name="q90",
        line=dict(color="#4575b4", width=0.5), showlegend=False,
        fill='tonexty', fillcolor='rgba(69,117,180,0.2)',
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=time_pred, y=q50, mode="lines", name="q50 中值",
        line=dict(color="#4575b4", width=2.2),
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=time_pred, y=actual, mode="lines", name="实际波形",
        line=dict(color="#d73027", width=1.8),
    ),
    secondary_y=False,
)

# Mark outside-band points
if outside.any():
    fig.add_trace(
        go.Scatter(
            x=time_pred[outside], y=actual[outside], mode="markers",
            name="越界点",
            marker=dict(color="red", size=8, symbol="x", line=dict(width=1.5, color="darkred")),
        ),
        secondary_y=False,
    )

# Add band label (invisible trace for legend)
fig.add_trace(
    go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="rgba(69,117,180,0.3)", width=8),
        name="[q10, q90] 置信带",
    ),
    secondary_y=False,
)

# Right Y: residual
fig.add_trace(
    go.Scatter(
        x=time_pred, y=residual, mode="lines", name="|actual − q50|",
        line=dict(color="orange", width=1.2, dash="dot"),
    ),
    secondary_y=True,
)

# Layout
fig.update_xaxes(title_text="时间 (秒, 0=预测起点)", zeroline=True, zerolinecolor="black",
                 zerolinewidth=0.8, showgrid=True, gridwidth=0.5, gridcolor="lightgray")
fig.update_yaxes(title_text="TS2 温度 (°C)", secondary_y=False, showgrid=True,
                 gridwidth=0.5, gridcolor="lightgray")
fig.update_yaxes(title_text="|残差| (°C)", secondary_y=True, showgrid=False)

fig.update_layout(
    title=dict(
        text=f"TS2 波形 · {type_label_map[type_choice]} · X=({row['X_valve']}, {row['X_pump']}, {row['X_accum']}) · "
             f"cooler_target={cooler_tgt} · cycle={cycle_choice}",
        font=dict(size=14),
    ),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    height=480,
    margin=dict(t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Verdict badge + metrics
# ============================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if verdict == "danger":
        st.error(f"⚠️ **危险 (DANGER)**")
    elif verdict == "warning":
        st.warning(f"⚠️ 预警 (WARNING)")
    else:
        st.success(f"✅ 正常 (NORMAL)")

with col2:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("s_level", f"{s_level:.4f} °C",
                  delta=f"阈值 τ={tau_level:.4f}" if s_level > tau_level else None)
    with c2:
        st.metric("s_shape", f"{s_shape:.4f}",
                  delta=f"阈值 τ={tau_shape:.4f}" if s_shape > tau_shape else None)

# ============================================================
# Bottom metrics (test set summary)
# ============================================================
st.markdown("---")
st.subheader("测试组汇总指标 (24 X, s_level/s_shape 分解)")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("检出率 (A 类)", f"{a_det:.4f}")
with m2:
    st.metric("近距误报率 (B 类)", f"{b_fpr:.4f}")
with m3:
    st.metric("远距误报率 (C 类)", f"{c_fpr:.4f}")

# ============================================================
# Method explanation
# ============================================================
with st.expander("方法说明"):
    st.markdown("""
    - **置信带**来自时序基础模型 Chronos-Bolt-Small 对健康波形 (cooler=100) 的零样本延拓，
      输入为同一工况下的前 5 个健康 cycle（共 300 个采样点），输出为后 60 个点的 q10/q50/q90 分位数预测。
    - **判决**由确定性 Python 规则执行：将实际波形的逐点残差分解为基线偏移 (s_level)
      与波形畸变 (s_shape)，与仅用健康数据标定的 P95 阈值比较，不经过任何 LLM。
    - **阈值**仅使用标定组（前 24 个工况配置 X）的远距健康对照窗口标定，
      故障数据从未参与阈值设定。
    """)

# ============================================================
# Footer: import list self-check
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("imports: streamlit, plotly, numpy, pandas")
