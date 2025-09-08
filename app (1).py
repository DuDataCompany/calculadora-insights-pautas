# app.py
# Streamlit app: Temperatura da Pauta + Forecast (CSV)
import math
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Temperatura da Pauta", page_icon="🔥", layout="wide")

# ----------------- Configurações padrão -----------------
DEFAULT_CONFIG = {
    "metas": {
        "mencoes": 500,        # Menções - 500
        "engajamentos": 6000,  # Engajamentos - 6000
        "sentimento": 70.0,    # Sentimento - meta >= 70 (0-100)
        "brandfit": 7.0        # Brandfit - meta >= 7 (0-10)
    },
    "pesos_iniciais": {
        "mencoes": 0.5,
        "engajamentos": 2.0,
        "sentimento": 1.0,
        "brandfit": 0.75
    },
    "caps": {"ratio_cap": 2.0},  # cap nas razões vs. meta
    "logistic_k": 4.0,           # inclinação da logística (score)
    "holdout_days": 28           # janela de avaliação dos modelos
}

# ----------------- Funções de Temperatura -----------------
def normalize_features(mencoes, engaj, sentimento, brandfit, config):
    metas = config["metas"]; caps = config["caps"]
    r_menc = min(max(0.0, mencoes / max(1.0, metas["mencoes"])), caps["ratio_cap"])
    r_eng  = min(max(0.0, engaj   / max(1.0, metas["engajamentos"])), caps["ratio_cap"])
    r_sent = min(max(0.0, sentimento / max(1e-6, metas["sentimento"])), caps["ratio_cap"])
    r_bfit = min(max(0.0, brandfit / max(1e-6, metas["brandfit"])), caps["ratio_cap"])

    menc_log = math.log1p(mencoes) / math.log1p(metas["mencoes"])
    eng_log  = math.log1p(engaj)   / math.log1p(metas["engajamentos"])

    sent_smooth = math.tanh(r_sent)
    bfit_smooth = math.tanh(r_bfit)

    return np.array([r_menc, menc_log, r_eng, eng_log, r_sent, sent_smooth, r_bfit, bfit_smooth], dtype=float)

def compute_composite(features, config):
    pesos = config["pesos_iniciais"]
    r_menc, menc_log, r_eng, eng_log, r_sent, sent_smooth, r_bfit, bfit_smooth = features

    comp_menc = 0.5*r_menc + 0.5*menc_log
    comp_eng  = 0.5*r_eng  + 0.5*eng_log
    comp_sent = 0.5*r_sent + 0.5*sent_smooth
    comp_bfit = 0.5*r_bfit + 0.5*bfit_smooth

    w = np.array([pesos["mencoes"], pesos["engajamentos"], pesos["sentimento"], pesos["brandfit"]], dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)

    composite = w[0]*comp_menc + w[1]*comp_eng + w[2]*comp_sent + w[3]*comp_bfit
    comps = {"mencoes": comp_menc, "engajamentos": comp_eng, "sentimento": comp_sent, "brandfit": comp_bfit}
    weights = {"mencoes": w[0], "engajamentos": w[1], "sentimento": w[2], "brandfit": w[3]}
    return composite, comps, weights

def composite_to_score(composite, config):
    k = config["logistic_k"]
    prob_like = 1.0 / (1.0 + math.exp(-k*(composite - 1.0)))
    return float(100.0 * prob_like)

def score_to_composite_target(score_target, config):
    k = config["logistic_k"]
    p = max(1e-6, min(1 - 1e-6, score_target/100.0))
    return 1.0 + (1.0/k) * math.log(p/(1.0 - p))

def classify(score):
    if score <= 40:  return "FRIA"
    if score <= 74:  return "MORNA"
    return "QUENTE"

def analyze_drivers(values, features, composite, comps, weights, score, config):
    metas = config["metas"]
    ratios = {"mencoes": features[0], "engajamentos": features[2], "sentimento": features[4], "brandfit": features[6]}
    contribs = {k: weights[k]*comps[k] for k in comps.keys()}
    contribs_sorted = sorted(contribs.items(), key=lambda x: x[1], reverse=True)

    ups   = [k for k,_ in contribs_sorted if ratios[k] >= 1.0]
    downs = [k for k,_ in contribs_sorted if ratios[k] < 1.0]

    recs = []
    if score < 75:
        composite_target = score_to_composite_target(75.0, config)
        delta_needed = max(0.0, composite_target - composite)
        gaps = [(k, (1.0 - min(1.0, ratios[k])), weights[k]) for k in ratios if ratios[k] < 1.0]
        gaps_sorted = sorted(gaps, key=lambda x: (x[1]*x[2]), reverse=True)
        for k, _, _ in gaps_sorted[:2]:
            if k in ["mencoes", "engajamentos"]:
                recs.append(f"- Aumente **{k}** até ~{int(metas[k]):,} (alvo de meta).".replace(",", "."))
            elif k == "sentimento":
                recs.append(f"- Eleve **sentimento** para ≥ {metas['sentimento']:.0f} via criativos de valência positiva.")
            elif k == "brandfit":
                recs.append(f"- Suba **brandfit** para ≥ {metas['brandfit']:.1f} alinhando mensagem e territórios de marca.")
        if delta_needed > 0:
            recs.append(f"- Ganho composto necessário ~{delta_needed:.2f} para chegar a score ≈ 75.")
    else:
        recs.append("- **Manter pilares ≥ meta** e escalar formatos/canais vencedores.")
        if ratios['brandfit'] < 1.0:
            recs.append("- **Ajustar brandfit**: refine narrativa/CTAs para aderir mais ao território da marca.")

    txt = []
    if ups:   txt.append("🔼 **Puxaram o score para cima:** " + ", ".join([u.capitalize() for u in ups]))
    if downs: txt.append("🔽 **Seguraram o score:** " + ", ".join([d.capitalize() for d in downs]))
    if recs:
        txt.append("🛠️ **Recomendações:**"); txt.extend(recs)
    return "\n".join(txt) if txt else "Sem destaques relevantes; pilares próximos da meta."

# ----------------- Gauge helpers -----------------
def render_gauge_plotly(score: float, title: str="Temperatura", height: int=360, width: int=640):
    # Cinza para barra
    bands = [(0, 40, "#D9534F"), (40, 70, "#F0AD4E"), (70, 100, "#5CB85C")]
    gauge_steps = [dict(range=[a, b], color=c) for (a, b, c) in bands]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(0, min(100, score)),
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#808080"},
            "steps": gauge_steps
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=height, width=width)
    return fig

def pct_of_goal(value, goal):
    try:
        return max(0.0, min(100.0, (float(value) / float(goal)) * 100.0))
    except Exception:
        return 0.0

# ----------------- Forecast helpers -----------------
def prepare_daily_series(df, date_col, value_col, dayfirst=True):
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], dayfirst=dayfirst, errors='coerce')
    s = s.dropna(subset=[date_col])
    s = s.sort_values(date_col)
    daily = s.groupby(pd.Grouper(key=date_col, freq="D"))[value_col].sum().reset_index()
    full_idx = pd.date_range(daily[date_col].min(), daily[date_col].max(), freq="D")
    daily = daily.set_index(date_col).reindex(full_idx).fillna(0.0).rename_axis(date_col).reset_index()
    daily[value_col] = daily[value_col].astype(float)
    return daily, date_col, value_col

def forecast_lr_with_dow(daily: pd.DataFrame, date_col: str, value_col: str, horizon_days: int):
    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    n = len(df)
    if horizon_days > 0:
        future_dates = pd.date_range(df[date_col].iloc[-1] + pd.Timedelta(days=1),
                                     periods=horizon_days, freq="D")
        future = pd.DataFrame({date_col: future_dates})
        all_df = pd.concat([df[[date_col]], future], ignore_index=True)
    else:
        all_df = df[[date_col]].copy()
    all_df["t"] = np.arange(len(all_df))
    all_df["dow"] = all_df[date_col].dt.dayofweek
    X_all = pd.get_dummies(all_df[["t", "dow"]], columns=["dow"], drop_first=True)
    X_hist = X_all.iloc[:n]
    y_hist = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
    lr = LinearRegression()
    lr.fit(X_hist.values, y_hist)
    hist_pred = lr.predict(X_hist.values)
    if horizon_days > 0:
        X_future = X_all.iloc[n:]
        y_future = lr.predict(X_future.values)
        y_future = np.maximum(0, y_future)
        fc = pd.DataFrame({date_col: all_df[date_col].iloc[n:], "pred": y_future})
    else:
        fc = pd.DataFrame(columns=[date_col, "pred"])
    return lr, hist_pred, fc

def forecast_holt_winters(daily: pd.DataFrame, date_col: str, value_col: str, horizon_days: int):
    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    ts = df.set_index(date_col)[value_col].asfreq("D")
    ts = pd.to_numeric(ts, errors="coerce").interpolate("time").fillna(method="bfill").fillna(method="ffill")
    if len(ts) >= 14:
        model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated")
    else:
        model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
    fit = model.fit(optimized=True)
    hist_pred = fit.fittedvalues.to_numpy()
    if horizon_days > 0:
        y_future = fit.forecast(horizon_days).to_numpy()
        y_future = np.maximum(0, y_future)
        future_dates = pd.date_range(df[date_col].iloc[-1] + pd.Timedelta(days=1),
                                     periods=horizon_days, freq="D")
        fc = pd.DataFrame({date_col: future_dates, "pred": y_future})
    else:
        fc = pd.DataFrame(columns=[date_col, "pred"])
    return fit, hist_pred, fc

def evaluate_models(daily: pd.DataFrame, date_col: str, value_col: str, holdout_days: int):
    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).reset_index(drop=True)
    n = len(df)
    if n <= holdout_days + 7:
        fit_hw, hist_hw, _ = forecast_holt_winters(df, date_col, value_col, horizon_days=0)
        return "holtwinters", {"fit": fit_hw, "hist_pred": hist_hw}
    split_idx = n - holdout_days
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    try:
        _, hist_lr, fc_lr = forecast_lr_with_dow(train, date_col, value_col, horizon_days=len(test))
        mape_lr = mean_absolute_percentage_error(test[value_col].to_numpy(), fc_lr["pred"].to_numpy())
    except Exception:
        mape_lr = np.inf
    try:
        _, hist_hw, fc_hw = forecast_holt_winters(train, date_col, value_col, horizon_days=len(test))
        mape_hw = mean_absolute_percentage_error(test[value_col].to_numpy(), fc_hw["pred"].to_numpy())
    except Exception:
        mape_hw = np.inf
    if mape_hw <= mape_lr:
        fit_full, hist_full, _ = forecast_holt_winters(df, date_col, value_col, horizon_days=0)
        return "holtwinters", {"fit": fit_full, "hist_pred": hist_full}
    else:
        lr_full, hist_full, _ = forecast_lr_with_dow(df, date_col, value_col, horizon_days=0)
        return "lr_dow", {"lr": lr_full, "hist_pred": hist_full}

def forecast_best_model(daily: pd.DataFrame, date_col: str, value_col: str, holdout_days: int, horizon_days: int):
    winner, art = evaluate_models(daily, date_col, value_col, holdout_days)
    if winner == "holtwinters":
        fit = art["fit"]
        hist_pred = art["hist_pred"]
        if horizon_days > 0:
            y_future = fit.forecast(horizon_days).to_numpy()
            y_future = np.maximum(0, y_future)
            future_dates = pd.date_range(daily[date_col].iloc[-1] + pd.Timedelta(days=1),
                                         periods=horizon_days, freq="D")
            fc = pd.DataFrame({date_col: future_dates, "pred": y_future})
        else:
            fc = pd.DataFrame(columns=[date_col, "pred"])
    else:
        lr, hist_pred, fc = forecast_lr_with_dow(daily, date_col, value_col, horizon_days=horizon_days)
    return winner, hist_pred, fc

def detect_peaks(forecast_df: pd.DataFrame, date_col: str, value_col: str = "pred", q: float = 0.90):
    if len(forecast_df) < 3:
        return []
    vals = forecast_df[value_col].to_numpy()
    thr = np.quantile(vals, q)
    peaks = []
    for i in range(1, len(vals)-1):
        if vals[i] > vals[i-1] and vals[i] > vals[i+1] and vals[i] >= thr:
            peaks.append((forecast_df[date_col].iloc[i], float(vals[i])))
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks

def summarize_trend(forecast_df: pd.DataFrame, value_col: str = "pred"):
    if len(forecast_df) == 0:
        return "sem forecast", 0.0
    t = np.arange(len(forecast_df)).reshape(-1, 1)
    lr = LinearRegression().fit(t, forecast_df[value_col].to_numpy())
    slope = float(lr.coef_[0])
    if slope > 0.05:
        label = "tendência de ALTA"
    elif slope < -0.05:
        label = "tendência de BAIXA"
    else:
        label = "tendência ESTÁVEL"
    return label, slope

# ----------------- UI -----------------
st.title("🔥 Temperatura da Pauta & Forecast")
st.caption("Creative Data · DuData — Streamlit App")

tab1, tab2 = st.tabs(["🌡️ Temperatura da Pauta", "📈 Forecast (CSV)"])

with tab1:
    st.subheader("Entradas da Pauta")
    with st.sidebar:
        st.markdown("### Metas")
        meta_m = st.number_input("Meta de Menções", min_value=1, value=DEFAULT_CONFIG["metas"]["mencoes"], step=50)
        meta_e = st.number_input("Meta de Engajamentos", min_value=1, value=DEFAULT_CONFIG["metas"]["engajamentos"], step=100)
        meta_s = st.number_input("Meta de Sentimento", min_value=1.0, value=float(DEFAULT_CONFIG["metas"]["sentimento"]), step=1.0)
        meta_b = st.number_input("Meta de Brandfit", min_value=0.1, max_value=10.0, value=float(DEFAULT_CONFIG["metas"]["brandfit"]), step=0.1)

        st.markdown("### Pesos (normalizados automaticamente)")
        pw_m = st.number_input("Peso: Menções", min_value=0.0, value=float(DEFAULT_CONFIG["pesos_iniciais"]["mencoes"]), step=0.1)
        pw_e = st.number_input("Peso: Engajamentos", min_value=0.0, value=float(DEFAULT_CONFIG["pesos_iniciais"]["engajamentos"]), step=0.1)
        pw_s = st.number_input("Peso: Sentimento", min_value=0.0, value=float(DEFAULT_CONFIG["pesos_iniciais"]["sentimento"]), step=0.1)
        pw_b = st.number_input("Peso: Brandfit", min_value=0.0, value=float(DEFAULT_CONFIG["pesos_iniciais"]["brandfit"]), step=0.1)

    config = {
        "metas": {"mencoes": meta_m, "engajamentos": meta_e, "sentimento": meta_s, "brandfit": meta_b},
        "pesos_iniciais": {"mencoes": pw_m, "engajamentos": pw_e, "sentimento": pw_s, "brandfit": pw_b},
        "caps": DEFAULT_CONFIG["caps"],
        "logistic_k": DEFAULT_CONFIG["logistic_k"],
        "holdout_days": DEFAULT_CONFIG["holdout_days"]
    }

    colA, colB, colC = st.columns([1.2, 1.2, 1])
    pauta = colA.text_input("Nome da pauta", value="Minha Pauta")
    menc_ig = colA.number_input("Menções no Instagram", min_value=0, value=0, step=10)
    menc_tw = colA.number_input("Menções no Twitter", min_value=0, value=0, step=10)

    eng_ig  = colB.number_input("Engajamentos no Instagram", min_value=0, value=0, step=50)
    eng_tw  = colB.number_input("Engajamentos no Twitter", min_value=0, value=0, step=50)

    brand   = colC.number_input("Brandfit (0-10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    sent    = colC.number_input("Sentimento (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

    menc_total = (menc_ig or 0) + (menc_tw or 0)
    eng_total  = (eng_ig or 0) + (eng_tw or 0)

    feats = normalize_features(menc_total, eng_total, sent, brand, config)
    composite, comps, weights = compute_composite(feats, config)
    score = composite_to_score(composite, config)
    classe = classify(score)
    values = {"mencoes": menc_total, "engajamentos": eng_total, "sentimento": sent, "brandfit": brand}
    analise = analyze_drivers(values, feats, composite, comps, weights, score, config)

    st.markdown(f"### Score final: **{score:.1f}** · Classe: **{classe}**")
    st.markdown("#### Análise de drivers")
    st.markdown(analise)

    # Gauge principal
    st.plotly_chart(render_gauge_plotly(score, title="Temperatura da Pauta"), use_container_width=True)

    # Gauges por variável (progresso vs meta)
    st.markdown("#### Progresso vs. Meta por variável")
    gauge_vals = {
        "Menções": pct_of_goal(menc_total, meta_m),
        "Engajamentos": pct_of_goal(eng_total, meta_e),
        "Sentimento": pct_of_goal(sent, meta_s),
        "Brandfit": pct_of_goal(brand, meta_b),
    }
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    for (nome, v), ax in zip(gauge_vals.items(), cols):
        ax.plotly_chart(render_gauge_plotly(v, title=nome, height=220, width=220), use_container_width=True)

with tab2:
    st.subheader("Forecast de Conversas (CSV)")
    horizon = st.number_input("Horizonte (dias)", min_value=1, max_value=730, value=180, step=1)
    up = st.file_uploader("Envie um CSV com colunas de data e valor", type=["csv"])
    dayfirst = st.checkbox("Datas no formato dia/mês/ano? (pt-BR)", value=True)
    if up is not None:
        df = pd.read_csv(up)
        st.write("Prévia do CSV:", df.head())

        # seleção de colunas
        cols = list(df.columns)
        date_col = st.selectbox("Coluna de data", options=cols, index=0)
        value_col = st.selectbox("Coluna de valor", options=cols, index=min(1, len(cols)-1))

        if st.button("Gerar Forecast"):
            daily, date_col, value_col = prepare_daily_series(df, date_col, value_col, dayfirst=dayfirst)
            winner, hist_pred, forecast_df = forecast_best_model(daily, date_col, value_col, DEFAULT_CONFIG["holdout_days"], horizon)

            total_h = float(forecast_df["pred"].sum())
            mean_h  = float(forecast_df["pred"].mean())
            med_h   = float(forecast_df["pred"].median())
            p10, p90 = np.percentile(forecast_df["pred"].values, [10, 90])

            trend_label, slope = summarize_trend(forecast_df, value_col="pred")
            peaks = detect_peaks(forecast_df, date_col=date_col, value_col="pred", q=0.90)

            st.markdown(f"**Modelo escolhido:** `{winner}`")
            st.metric("Soma prevista", f"{total_h:,.0f}".replace(",", "."))
            st.metric("Média diária prevista", f"{mean_h:.1f}")
            st.write(f"**Mediana:** {med_h:.1f} · **P10–P90:** {p10:.0f} a {p90:.0f} conversas/dia")
            st.write(f"**Tendência:** {trend_label} (slope={slope:.3f} conversas/dia)")

            # Tabela
            forecast_table = forecast_df.rename(columns={date_col: "data", "pred": "conversas_previstas"})
            forecast_table["conversas_previstas"] = forecast_table["conversas_previstas"].round(2)
            st.dataframe(forecast_table, use_container_width=True)

            # Download CSV
            csv_bytes = forecast_table.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar CSV de previsões", data=csv_bytes, file_name=f"forecast_{value_col}_{horizon}d.csv", mime="text/csv")

            # Gráfico (Histórico x Ajuste x Previsão) via Matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(daily[date_col], daily[value_col], linewidth=2, label="Histórico")
            ax.plot(daily[date_col], hist_pred, linestyle="--", linewidth=1.6, label="Ajuste (hist.)")
            ax.plot(forecast_df[date_col], forecast_df["pred"], linewidth=2.2, label=f"Previsão (+{horizon}d)")
            ax.axvspan(daily[date_col].iloc[-1], forecast_df[date_col].iloc[-1], alpha=0.08)
            for d, v in peaks[:5]:
                ax.scatter(pd.to_datetime(d), v, s=36, zorder=5)
                ax.annotate(f\"{v:.0f}\", (pd.to_datetime(d), v), textcoords=\"offset points\", xytext=(0,8), ha=\"center\", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.set_title(\"Tendência da Conversa — Forecast\") 
            ax.set_xlabel(\"Data\"); ax.set_ylabel(value_col); ax.legend()
            st.pyplot(fig)