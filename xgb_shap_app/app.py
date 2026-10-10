"""Streamlit GUI for the XGBoost + TreeSHAP engine.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

import engine as eng

st.set_page_config(page_title="XGBoost + TreeSHAP", page_icon="🌲", layout="wide")

MAX_SHAP_PLOT_ROWS = 2000
MAX_INTERACTION_ROWS = 300


def show_fig(fig=None) -> None:
    st.pyplot(fig or plt.gcf(), clear_figure=True, width="stretch")
    plt.close("all")


@st.cache_data(show_spinner=False)
def cached_load(data: bytes, name: str, sheet: str | None) -> pd.DataFrame:
    import io

    return eng.load_table(io.BytesIO(data), name, sheet)


@st.cache_data(show_spinner=False)
def cached_sheets(data: bytes, name: str) -> list[str]:
    import io

    return eng.list_sheets(io.BytesIO(data), name)


# --------------------------------------------------------------------------- #
# Sidebar: data + configuration
# --------------------------------------------------------------------------- #
st.sidebar.title("🌲 XGBoost + TreeSHAP")
upload = st.sidebar.file_uploader("Training data (.csv / .xlsx)", type=["csv", "xlsx", "xlsm", "xls"])

LANDING = """
Upload a **.csv** or **.xlsx** file in the sidebar to train a model, or open the
**🔮 Predict** tab to score data with a model you exported earlier (.json).

**What you get**
- Automatic task detection (regression / binary / multiclass) and type handling
  (native categorical support, date parsing, missing values handled by XGBoost)
- Train / validation / test split with early stopping and tunable hyperparameters
- Performance metrics, learning curves, ROC, confusion matrix, residuals
- **Exact TreeSHAP** explanations: global importance, beeswarm, dependence plots,
  per-row waterfall explanations, and pairwise SHAP interaction values
- Score new .csv / .xlsx files and download predictions with per-row SHAP values
"""


def training_sidebar(upload) -> tuple[pd.DataFrame, str | None, set[str]] | None:
    """Sidebar controls for training. Returns (df, sheet, auto_drop) or None."""
    raw = upload.getvalue()
    sheets = cached_sheets(raw, upload.name)
    sheet = st.sidebar.selectbox("Sheet", sheets) if len(sheets) > 1 else (sheets[0] if sheets else None)

    try:
        df = cached_load(raw, upload.name, sheet)
    except Exception as e:  # noqa: BLE001 — show any parse error to the user
        st.error(f"Could not read file: {e}")
        return None

    if df.shape[1] < 2:
        st.error("Need at least two columns (features + target).")
        return None

    cols = list(df.columns)
    target = st.sidebar.selectbox("Target column", cols, index=len(cols) - 1)
    suggested_task = eng.infer_task(df[target])
    task = st.sidebar.selectbox("Task", eng.TASKS, index=eng.TASKS.index(suggested_task))

    candidates = [c for c in cols if c != target]
    auto_drop = set(eng.suggest_drop_columns(df[candidates]))
    features = st.sidebar.multiselect(
        "Features",
        candidates,
        default=[c for c in candidates if c not in auto_drop],
        help="ID-like columns (unique per row) are deselected by default.",
    )
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = st.sidebar.multiselect(
        "Treat as categorical",
        numeric_features,
        help="Numeric columns that are really codes, e.g. store IDs, zip codes or product codes. "
             "Text columns are already treated as categories.",
    )
    too_many = {c: n for c in categorical if (n := df[c].nunique()) > eng.MAX_CATEGORIES}
    for c, n in too_many.items():
        st.sidebar.warning(f"`{c}` has {n:,} distinct values. Only the {eng.MAX_CATEGORIES} most common "
                           "are kept as categories; the rest are treated as missing.")

    with st.sidebar.expander("Hyperparameters", expanded=False):
        d = eng.DEFAULT_PARAMS
        params = dict(
            n_estimators=st.number_input("n_estimators (max trees)", 10, 20000, d["n_estimators"], 50),
            learning_rate=st.number_input("learning_rate", 0.001, 1.0, d["learning_rate"], 0.01, format="%.3f"),
            max_depth=st.slider("max_depth", 1, 16, d["max_depth"]),
            min_child_weight=st.number_input("min_child_weight", 0.0, 100.0, d["min_child_weight"], 0.5),
            subsample=st.slider("subsample", 0.1, 1.0, d["subsample"], 0.05),
            colsample_bytree=st.slider("colsample_bytree", 0.1, 1.0, d["colsample_bytree"], 0.05),
            gamma=st.number_input("gamma (min split loss)", 0.0, 100.0, d["gamma"], 0.1),
            reg_alpha=st.number_input("reg_alpha (L1)", 0.0, 100.0, d["reg_alpha"], 0.1),
            reg_lambda=st.number_input("reg_lambda (L2)", 0.0, 100.0, d["reg_lambda"], 0.1),
        )

    with st.sidebar.expander("Training options", expanded=False):
        test_size = st.slider("Test fraction", 0.05, 0.5, 0.2, 0.05)
        use_es = st.checkbox("Early stopping", True)
        es_rounds = st.number_input("Early-stopping rounds", 5, 1000, 50, 5, disabled=not use_es)
        val_size = st.slider("Validation fraction (of train)", 0.05, 0.4, 0.15, 0.05, disabled=not use_es)
        balance = st.checkbox("Balance classes", False, disabled=task == eng.REGRESSION)
        seed = st.number_input("Random seed", 0, 2**31 - 1, 42)
        use_gpu = st.checkbox("Use GPU (CUDA)", False)

    train_clicked = st.sidebar.button("🚀 Train model", type="primary", width="stretch")

    if train_clicked:
        if not features:
            st.sidebar.error("Select at least one feature.")
        else:
            with st.spinner("Training XGBoost…"):
                try:
                    st.session_state["result"] = eng.train(
                        df,
                        target=target,
                        features=features,
                        task=task,
                        params=params,
                        test_size=test_size,
                        val_size=val_size,
                        early_stopping_rounds=int(es_rounds) if use_es else None,
                        balance_classes=balance,
                        seed=int(seed),
                        use_gpu=use_gpu,
                        categorical=categorical,
                    )
                    st.session_state["result_source"] = upload.name
                except Exception as e:  # noqa: BLE001
                    st.session_state.pop("result", None)
                    st.error(f"Training failed: {e}")

    return df, sheet, auto_drop


loaded = training_sidebar(upload) if upload is not None else None
df, sheet, auto_drop = loaded if loaded else (None, None, set())

result: eng.TrainResult | None = st.session_state.get("result")

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
tabs = st.tabs(
    ["📄 Data", "📈 Performance", "🌐 Global SHAP", "🔗 Dependence", "🔍 Local explanation",
     "🧩 Interactions", "🔮 Predict", "💾 Export"]
)

with tabs[0]:
    if df is None:
        st.title("XGBoost engine with TreeSHAP explanations")
        st.markdown(LANDING)
    else:
        st.subheader(f"{upload.name}" + (f" — {sheet}" if sheet else ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.dataframe(df.head(200), width="stretch")
        summary = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "unique": df.nunique(),
                "missing %": (df.isna().mean() * 100).round(1),
            }
        )
        st.markdown("**Column summary**")
        st.dataframe(summary, width="stretch")
        if auto_drop:
            st.info(f"Auto-deselected ID-like columns: {', '.join(sorted(auto_drop))}")

# --- Predict --------------------------------------------------------------- #
def excel_note(n_rows: int) -> str:
    if n_rows > eng.EXCEL_MAX_ROWS:
        sheets = -(-n_rows // eng.EXCEL_MAX_ROWS)
        return f"{n_rows:,} rows exceed Excel's sheet limit, so the workbook is split across {sheets} sheets."
    if n_rows > 200_000:
        return f"Building an Excel file with {n_rows:,} rows can take a few minutes. CSV is much faster."
    return ""


SESSION_MODEL = "Model trained in this session"
SAVED_MODEL = "Load a saved model (.json)"


def pick_scoring_model(session_result: eng.TrainResult | None) -> eng.Predictor | None:
    options = [SESSION_MODEL, SAVED_MODEL] if session_result is not None else [SAVED_MODEL]
    source = st.radio("Model", options, horizontal=True, key="model_source")
    if source == SESSION_MODEL:
        return session_result

    model_file = st.file_uploader("Model file (.json exported from this app, or any XGBoost model)",
                                  type=["json", "ubj"], key="model_file")
    if model_file is None:
        return None
    cached = st.session_state.get("loaded_model")
    if cached is None or cached[0] != model_file.file_id:
        try:
            st.session_state["loaded_model"] = (model_file.file_id, eng.load_model(model_file.getvalue()))
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("loaded_model", None)
            st.error(f"Could not load model: {e}")
            return None
    model = st.session_state["loaded_model"][1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task", model.task.split()[0].title())
    c2.metric("Target", model.target)
    c3.metric("Features", len(model.feature_names))
    c4.metric("Trees used", model.n_trees_used)
    if model.class_names:
        st.caption(f"Classes: {', '.join(model.class_names)}")
    for note in model.notes:
        st.warning(note)
    return model


def render_predict(session_result: eng.TrainResult | None) -> None:
    st.subheader("Score new data")
    model = pick_scoring_model(session_result)
    if model is None:
        return
    st.caption(f"File must contain the feature columns: {', '.join(model.feature_names)}")
    new = st.file_uploader("New data (.csv / .xlsx)", type=["csv", "xlsx", "xlsm", "xls"], key="new")
    if new is None:
        return
    nraw = new.getvalue()
    nsheets = cached_sheets(nraw, new.name)
    nsheet = st.selectbox("Sheet", nsheets, key="nsheet") if len(nsheets) > 1 else (nsheets[0] if nsheets else None)
    include_shap = st.checkbox("Include per-row SHAP values", True)
    # Score once per (model, file, sheet, option) rather than on every rerun
    score_key = (id(model), new.file_id, nsheet, include_shap)
    if st.session_state.get("scored_key") != score_key:
        try:
            with st.spinner("Scoring…"):
                ndf = cached_load(nraw, new.name, nsheet)
                scored = eng.shap_frame(model, ndf) if include_shap else model.predict_frame(ndf)
                st.session_state["scored"] = pd.concat(
                    [ndf.reset_index(drop=True), scored.reset_index(drop=True)], axis=1
                )
                st.session_state["scored_key"] = score_key
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("scored_key", None)
            st.error(f"Scoring failed: {e}")
            return
    out = st.session_state["scored"]
    st.success(f"Scored {len(out):,} rows.")
    st.dataframe(out.head(500), width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download CSV", lambda: out.to_csv(index=False).encode(), "predictions.csv",
                       "text/csv", width="stretch")
    c2.download_button("⬇️ Download Excel", lambda: eng.to_excel_bytes({"predictions": out}),
                       "predictions.xlsx", width="stretch")
    if note := excel_note(len(out)):
        st.caption(note)


with tabs[6]:
    render_predict(result)

if result is None:
    hint = ("Configure the model in the sidebar and click **Train model**." if df is not None
            else "Upload training data in the sidebar to train a model.")
    for i, t in enumerate(tabs[1:], 1):
        if i != 6:
            with t:
                st.info(hint)
    st.stop()

if upload is not None and st.session_state.get("result_source") != upload.name:
    st.warning("The trained model came from a different file. Retrain to refresh.")

# Shared SHAP data (test set, capped)
X_explain = result.X_test
if len(X_explain) > MAX_SHAP_PLOT_ROWS:
    X_explain = X_explain.sample(MAX_SHAP_PLOT_ROWS, random_state=0)

is_multi = result.task == eng.MULTICLASS
class_names = result.class_names


def class_picker(key: str) -> int:
    if not is_multi:
        return 0
    name = st.selectbox("Explain class", class_names, key=key)
    return class_names.index(name)


space = "log-odds" if result.task != eng.REGRESSION else "target units"

# --- Performance ----------------------------------------------------------- #
with tabs[1]:
    st.subheader(f"{result.task.title()} — target: `{result.target}`")
    mcols = st.columns(len(result.metrics))
    for c, (k, v) in zip(mcols, result.metrics.items()):
        c.metric(k, f"{int(v)}" if k == "Trees used" else f"{v:.4f}")
    st.caption(
        f"Train {len(result.X_train):,} rows · Test {len(result.X_test):,} rows "
        "(metrics are on the held-out test set)."
    )

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, vals in result.history.items():
            fig.add_trace(go.Scatter(y=vals, mode="lines", name=name))
        best = result.best_iteration
        if best is not None:
            fig.add_vline(x=best, line_dash="dash", annotation_text="best")
        fig.update_layout(title=f"Learning curve ({result.eval_metric})", xaxis_title="Boosting round",
                          height=420, margin=dict(t=50, b=40))
        st.plotly_chart(fig, width="stretch")

    with right:
        if result.task == eng.REGRESSION:
            pred = result.test_pred
            fig = px.scatter(x=result.y_test, y=pred, opacity=0.6,
                             labels={"x": "Actual", "y": "Predicted"}, title="Predicted vs actual")
            lo, hi = float(min(result.y_test.min(), pred.min())), float(max(result.y_test.max(), pred.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal",
                                     line=dict(dash="dash")))
            fig.update_layout(height=420, margin=dict(t=50, b=40))
            st.plotly_chart(fig, width="stretch")
        else:
            proba = result.test_pred
            cm = confusion_matrix(result.y_test, proba.argmax(1), labels=range(len(class_names)))
            fig = px.imshow(cm, x=class_names, y=class_names, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion matrix")
            fig.update_layout(height=420, margin=dict(t=50, b=40))
            st.plotly_chart(fig, width="stretch")

    if result.task == eng.BINARY:
        proba = result.test_pred[:, 1]
        fpr, tpr, _ = roc_curve(result.y_test, proba)
        fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="chance"))
        fig.update_layout(title=f"ROC curve (positive class = {class_names[1]})",
                          xaxis_title="False positive rate", yaxis_title="True positive rate", height=400)
        st.plotly_chart(fig, width="stretch")
    elif result.task == eng.REGRESSION:
        resid = result.y_test - result.test_pred
        fig = px.histogram(x=resid, nbins=50, title="Residuals (actual − predicted)", labels={"x": "Residual"})
        st.plotly_chart(fig, width="stretch")

# --- Global SHAP ----------------------------------------------------------- #
with tabs[2]:
    st.subheader("Global feature importance (TreeSHAP)")
    st.caption(f"Computed on {len(X_explain):,} test rows. SHAP values are in {space}.")
    imp = eng.global_importance(result, X_explain)
    left, right = st.columns([1, 1])
    with left:
        top = imp.head(25).iloc[::-1]
        fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h",
                     title="Mean |SHAP value|" + (" (summed over classes)" if is_multi else ""),
                     labels={"mean_abs_shap": "mean |SHAP|", "feature": ""})
        fig.update_layout(height=max(350, 22 * len(top) + 100), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.dataframe(imp, width="stretch", height=min(800, 36 * len(imp) + 40))
        st.caption("`gain` is XGBoost's built-in split-gain importance, shown for comparison.")

    st.markdown("**Beeswarm** — each dot is a row; colour = feature value (high → red)")
    k = class_picker("bee_class")
    expl = result.explanation(X_explain, k)
    max_display = st.slider("Features to show", 5, max(5, len(result.feature_names)),
                            min(15, len(result.feature_names)), key="bee_n") \
        if len(result.feature_names) > 5 else len(result.feature_names)
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    show_fig()

    if is_multi:
        st.markdown("**Per-class importance**")
        values, _ = result.shap_values(X_explain)
        per_class = pd.DataFrame(np.abs(values).mean(axis=0).T, index=result.feature_names, columns=class_names)
        per_class = per_class.loc[imp["feature"].head(20)]
        fig = px.bar(per_class.iloc[::-1], orientation="h", labels={"value": "mean |SHAP|", "index": ""})
        fig.update_layout(height=max(350, 22 * len(per_class) + 100), barmode="stack")
        st.plotly_chart(fig, width="stretch")

# --- Dependence ------------------------------------------------------------ #
with tabs[3]:
    st.subheader("SHAP dependence")
    st.caption("How a feature's value drives its contribution. Colour by a second feature to reveal interactions.")
    ordered = list(eng.global_importance(result, X_explain)["feature"])
    c1, c2, c3 = st.columns(3)
    feat = c1.selectbox("Feature", ordered, key="dep_feat")
    color_opt = c2.selectbox("Colour by", ["(auto: strongest interaction)", "(none)"] + ordered, key="dep_col")
    with c3:
        k = class_picker("dep_class")
    expl = result.explanation(X_explain, k)
    if color_opt == "(none)":
        shap.plots.scatter(expl[:, feat], show=False)
    elif color_opt.startswith("(auto"):
        # shap's interaction search correlates chunks that can be constant; 0/0 there is harmless
        with np.errstate(invalid="ignore", divide="ignore"):
            shap.plots.scatter(expl[:, feat], color=expl, show=False)
    else:
        shap.plots.scatter(expl[:, feat], color=expl[:, color_opt], show=False)
    if feat in result.spec.categories:
        cats = result.spec.categories[feat]
        if len(cats) <= 30:
            plt.xticks(range(len(cats)), cats, rotation=45, ha="right")
    show_fig()

# --- Local explanation ----------------------------------------------------- #
with tabs[4]:
    st.subheader("Explain a single prediction")
    source = st.radio("Row source", ["Test set", "Training set"], horizontal=True)
    X_src = result.X_test if source == "Test set" else result.X_train
    y_src = result.y_test if source == "Test set" else result.y_train
    pos = st.number_input(f"Row position (0 – {len(X_src) - 1})", 0, len(X_src) - 1, 0)
    row = X_src.iloc[[pos]]

    actual = y_src[pos]
    if result.task == eng.REGRESSION:
        pred_txt = f"{float(result.predict_raw(row)[0]):.4g}"
        actual_txt = f"{actual:.4g}"
    else:
        proba = result.predict_raw(row)[0]
        top = int(proba.argmax())
        pred_txt = f"{class_names[top]} (p = {proba[top]:.3f})"
        actual_txt = class_names[int(actual)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Original index", str(row.index[0]))
    c2.metric("Prediction", pred_txt)
    c3.metric("Actual", actual_txt)

    k = class_picker("local_class")
    expl = result.explanation(row, k)[0]
    max_disp = st.slider("Features to show", 3, max(3, len(result.feature_names)),
                         min(12, len(result.feature_names)), key="wf_n") \
        if len(result.feature_names) > 3 else len(result.feature_names)
    shap.plots.waterfall(expl, max_display=max_disp, show=False)
    show_fig()
    st.caption(
        f"Starts at the expected model output E[f(X)] and adds each feature's SHAP contribution "
        f"to reach this row's output f(x), in {space}."
    )

    contrib = pd.DataFrame(
        {"feature": result.feature_names, "value": [str(v) for v in expl.display_data], "shap": expl.values}
    ).sort_values("shap", key=np.abs, ascending=False)
    st.dataframe(contrib, width="stretch", hide_index=True)

    with st.expander("Force plot"):
        nonzero = np.abs(expl.values) > 0
        if not nonzero.any():
            # shap's force plot divides by the total |effect|, which is 0 here
            st.info("Every SHAP value for this row is zero: the prediction equals the base value, "
                    "so there is nothing to draw.")
        else:
            # shap divides by each side's span, which is 0 when a side has one feature;
            # the result (inf) only means "always label", so the warning is noise.
            with np.errstate(divide="ignore", invalid="ignore"):
                shap.plots.force(expl.base_values, expl.values[nonzero], features=np.round(expl.data[nonzero], 4),
                                 feature_names=list(np.array(result.feature_names)[nonzero]),
                                 matplotlib=True, show=False)
            show_fig()

# --- Interactions ---------------------------------------------------------- #
with tabs[5]:
    st.subheader("SHAP interaction values")
    st.caption(
        f"Pairwise TreeSHAP interaction effects (mean |value|), computed on up to "
        f"{MAX_INTERACTION_ROWS} test rows. Cost grows with features², so this runs on demand."
    )
    k = class_picker("int_class")
    if st.button("Compute interactions"):
        Xi = result.X_test.head(MAX_INTERACTION_ROWS)
        with st.spinner("Computing interaction values…"):
            inter = result.shap_interactions(Xi)
        if inter.ndim == 4:
            inter = inter[:, k]
        st.session_state["interactions"] = (k, np.abs(inter).mean(axis=0))

    stored = st.session_state.get("interactions")
    if stored is not None and stored[0] == k and stored[1].shape[0] == len(result.feature_names):
        mat = stored[1]
        names = result.feature_names
        order = np.argsort(-mat.sum(axis=1))[:20]
        sub = pd.DataFrame(mat[np.ix_(order, order)], index=[names[i] for i in order],
                           columns=[names[i] for i in order])
        fig = px.imshow(sub, color_continuous_scale="Viridis", aspect="auto",
                        title="Mean |interaction| (diagonal = main effect)")
        fig.update_layout(height=max(450, 28 * len(order) + 150))
        st.plotly_chart(fig, width="stretch")

        off = mat.copy()
        np.fill_diagonal(off, 0)
        iu = np.triu_indices_from(off, 1)
        pairs = pd.DataFrame(
            {"feature A": [names[i] for i in iu[0]], "feature B": [names[j] for j in iu[1]],
             "mean |interaction| ×2": 2 * off[iu]}
        ).sort_values("mean |interaction| ×2", ascending=False).head(25)
        st.markdown("**Strongest pairwise interactions**")
        st.dataframe(pairs, width="stretch", hide_index=True)

# --- Export ---------------------------------------------------------------- #
REPORT_SHAP_ROWS = 100_000

with tabs[7]:
    st.subheader("Export")
    st.download_button("⬇️ XGBoost model (.json)", eng.model_json_bytes(result), "xgb_model.json",
                       "application/json")
    st.caption("Reload it in the **🔮 Predict** tab to score new data, or in Python with "
               "`xgb.Booster(model_file='xgb_model.json')`. Class labels, category lists and date "
               "columns are stored inside the file.")

    def test_shap_table(max_rows: int | None = None) -> pd.DataFrame:
        X = result.X_test
        if max_rows is not None and len(X) > max_rows:
            X = X.sample(max_rows, random_state=0).sort_index()
        shap_df = eng.shap_frame(result, X, transformed=True)
        return pd.concat([X.reset_index(), shap_df.reset_index(drop=True)], axis=1)

    def report_xlsx() -> bytes:
        return eng.to_excel_bytes({
            "metrics": pd.DataFrame(result.metrics.items(), columns=["metric", "value"]),
            "importance": eng.global_importance(result, X_explain),
            "test_shap": test_shap_table(REPORT_SHAP_ROWS),
            "params": pd.DataFrame(result.params.items(), columns=["param", "value"]),
        })

    n_test = len(result.X_test)
    st.download_button("⬇️ SHAP report (.xlsx)", report_xlsx, "shap_report.xlsx")
    st.caption(
        "Metrics, importance, parameters, and test-set SHAP values"
        + (f" for a random sample of {REPORT_SHAP_ROWS:,} of {n_test:,} test rows." if n_test > REPORT_SHAP_ROWS
           else f" for all {n_test:,} test rows.")
    )
    st.download_button("⬇️ Full test-set SHAP values (.csv)",
                       lambda: test_shap_table().to_csv(index=False).encode(), "test_shap.csv", "text/csv")
    st.caption(f"All {n_test:,} test rows with predictions and per-feature SHAP values.")
