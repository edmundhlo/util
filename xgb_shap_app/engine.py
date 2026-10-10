"""XGBoost training engine with TreeSHAP explanations.

SHAP values are computed with XGBoost's native TreeSHAP implementation
(``pred_contribs`` / ``pred_interactions``), which is exact, fast, and
supports native categorical features. Values are in the model's margin
space (raw score for regression, log-odds for classification).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

REGRESSION = "regression"
BINARY = "binary classification"
MULTICLASS = "multiclass classification"
TASKS = (REGRESSION, BINARY, MULTICLASS)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def list_sheets(source: Any, filename: str) -> list[str]:
    """Return sheet names for an Excel file, or [] for CSV."""
    if Path(filename).suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        _rewind(source)
        return pd.ExcelFile(source).sheet_names
    return []


def load_table(source: Any, filename: str, sheet: str | None = None) -> pd.DataFrame:
    """Load a .csv or .xlsx file (path or file-like object) into a DataFrame."""
    suffix = Path(filename).suffix.lower()
    _rewind(source)
    if suffix == ".csv":
        df = pd.read_csv(source, sep=None, engine="python")  # sniff delimiter
    elif suffix in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(source, sheet_name=sheet or 0)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .csv or .xlsx.")
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(axis=1, how="all")


def _rewind(source: Any) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def infer_task(y: pd.Series) -> str:
    y = y.dropna()
    n_unique = y.nunique()
    if n_unique <= 2:
        return BINARY
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return MULTICLASS
    if pd.api.types.is_integer_dtype(y) or np.allclose(y, y.round()):
        if n_unique <= 10:
            return MULTICLASS
    return REGRESSION


def suggest_drop_columns(df: pd.DataFrame) -> list[str]:
    """Columns that look like IDs or free text (unique per row, non-numeric)."""
    out = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        if s.nunique(dropna=True) >= 0.95 * n and n > 20:
            if not pd.api.types.is_float_dtype(s):
                out.append(c)
    return out


MAX_CATEGORIES = 200  # per column; rarer categories are treated as missing


@dataclass
class FeatureSpec:
    """Remembers how raw columns were converted, so new data matches."""

    columns: list[str]
    categories: dict[str, list[Any] | None] = field(default_factory=dict)
    datetimes: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls, X: pd.DataFrame, max_categories: int = MAX_CATEGORIES, force_categorical: Iterable[str] = ()
    ) -> "FeatureSpec":
        """Infer column handling. ``force_categorical`` columns are treated as
        categories even if numeric (e.g. store IDs, zip codes, product codes)."""
        spec = cls(columns=list(X.columns))
        forced = set(force_categorical)
        for c in X.columns:
            s = X[c]
            if c in forced:
                cats = _category_labels(s).value_counts()
                spec.categories[c] = list(cats.index[:max_categories])
            elif pd.api.types.is_datetime64_any_dtype(s):
                spec.datetimes.append(c)
            elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                continue
            else:
                parsed = _try_parse_datetime(s)
                if parsed is not None:
                    spec.datetimes.append(c)
                    continue
                cats = _category_labels(s).value_counts()
                spec.categories[c] = list(cats.index[:max_categories])
        return spec

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        X = df[self.columns].copy()
        for c in self.columns:
            if c in self.datetimes:
                X[c] = pd.to_datetime(X[c], errors="coerce")
                X[c] = X[c].astype("int64").where(X[c].notna()) / 86_400e9  # days
                X[c] = X[c].astype(float)
            elif c in self.categories:
                cats = self.categories[c]
                if cats is not None and not all(isinstance(v, str) for v in cats):
                    X[c] = pd.Categorical(pd.to_numeric(X[c], errors="coerce"), categories=cats)
                else:
                    X[c] = pd.Categorical(_category_labels(X[c]), categories=cats)
            elif pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype(float)
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X


def _category_labels(s: pd.Series) -> pd.Series:
    """String labels for category values, missing kept as NaN.

    Whole numbers are written without a decimal point, so a code column that
    loads as float in one file (because it has blanks) and int in another
    still maps 3.0 and 3 to the same category "3".
    """
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        f = s.astype(float)
        out = f.astype(str)
        whole = f.notna() & (f % 1 == 0)
        out[whole] = f[whole].astype("int64").astype(str)
        return out.where(f.notna())
    return s.where(s.isna(), s.astype(str))


def _try_parse_datetime(s: pd.Series) -> pd.Series | None:
    sample = s.dropna().astype(str).head(200)
    if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").all():
        return None
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return None
    return parsed if parsed.notna().mean() > 0.9 else None


def numeric_view(X: pd.DataFrame) -> pd.DataFrame:
    """Categoricals -> integer codes (NaN for missing); used for SHAP colouring."""
    out = X.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            codes = out[c].cat.codes.astype(float)
            out[c] = codes.where(codes >= 0)
    return out.astype(float)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
DEFAULT_PARAMS: dict[str, Any] = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
)


@dataclass(kw_only=True)
class Predictor:
    """A fitted booster plus everything needed to score and explain raw data.

    Built by ``train`` or restored from a JSON file by ``load_model``.
    """

    booster: xgb.Booster
    task: str
    target: str
    spec: FeatureSpec
    class_names: list[str] = field(default_factory=list)
    best_iteration: int | None = None
    notes: list[str] = field(default_factory=list)  # caveats from loading

    @property
    def feature_names(self) -> list[str]:
        return self.spec.columns

    def _iteration_range(self) -> tuple[int, int]:
        best = self.best_iteration
        return (0, best + 1) if best is not None else (0, 0)

    @property
    def n_trees_used(self) -> int:
        return self._iteration_range()[1] or self.booster.num_boosted_rounds()

    # --- prediction -------------------------------------------------------- #
    def _dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X, enable_categorical=True)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Regression: predictions (n,). Classification: class probabilities (n, k)."""
        dm, it = self._dmatrix(X), self._iteration_range()
        if self.task == REGRESSION:
            return self.booster.predict(dm, iteration_range=it)
        # Apply the link to the margin ourselves so any classification objective
        # (logistic / logitraw, softprob / softmax) yields probabilities.
        margin = self.booster.predict(dm, output_margin=True, iteration_range=it)
        if self.task == BINARY:
            p = 1.0 / (1.0 + np.exp(-margin.reshape(-1)))
            return np.column_stack([1 - p, p])
        margin = margin - margin.max(axis=1, keepdims=True)
        e = np.exp(margin)
        return e / e.sum(axis=1, keepdims=True)

    def predict_frame(self, df: pd.DataFrame, transformed: bool = False) -> pd.DataFrame:
        X = df if transformed else self.spec.transform(df)
        out = pd.DataFrame(index=df.index)
        if self.task == REGRESSION:
            out["prediction"] = self.predict_raw(X)
        else:
            proba = self.predict_raw(X)
            out["prediction"] = np.asarray(self.class_names, dtype=object)[proba.argmax(axis=1)]
            for i, name in enumerate(self.class_names):
                out[f"proba_{name}"] = proba[:, i]
        return out

    # --- TreeSHAP ---------------------------------------------------------- #
    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Exact TreeSHAP.

        Returns (values, base_values):
          regression / binary: (n, f), (n,)
          multiclass:          (n, k, f), (n, k)
        """
        contribs = self.booster.predict(
            self._dmatrix(X), pred_contribs=True, iteration_range=self._iteration_range()
        )
        return contribs[..., :-1], contribs[..., -1]

    def shap_interactions(self, X: pd.DataFrame) -> np.ndarray:
        """TreeSHAP interaction values, (n, f, f) — or (n, k, f, f) for multiclass."""
        inter = self.booster.predict(
            self._dmatrix(X), pred_interactions=True, iteration_range=self._iteration_range()
        )
        return inter[..., :-1, :-1]

    def explanation(self, X: pd.DataFrame, class_index: int = 0) -> shap.Explanation:
        """Build a shap.Explanation (for one class if multiclass)."""
        values, base = self.shap_values(X)
        if values.ndim == 3:
            values, base = values[:, class_index, :], base[:, class_index]
        return shap.Explanation(
            values=values,
            base_values=base,
            data=numeric_view(X).values,
            display_data=_display_values(X),
            feature_names=self.feature_names,
        )


@dataclass(kw_only=True)
class TrainResult(Predictor):
    model: xgb.XGBModel
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    metrics: dict[str, float]
    history: dict[str, list[float]]
    eval_metric: str
    params: dict[str, Any]
    _test_pred: np.ndarray | None = field(default=None, repr=False)

    @property
    def test_pred(self) -> np.ndarray:
        """Cached predict_raw on the test set."""
        if self._test_pred is None:
            self._test_pred = self.predict_raw(self.X_test)
        return self._test_pred


def _display_values(X: pd.DataFrame) -> np.ndarray:
    disp = X.astype(object).copy()
    for c in disp.columns:
        if pd.api.types.is_float_dtype(X[c]):
            disp[c] = X[c].map(lambda v: v if pd.isna(v) else float(f"{v:.4g}"))
    return disp.values


def train(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    task: str,
    params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    val_size: float = 0.15,
    early_stopping_rounds: int | None = 50,
    balance_classes: bool = False,
    seed: int = 42,
    use_gpu: bool = False,
    categorical: Iterable[str] = (),
) -> TrainResult:
    params = {**DEFAULT_PARAMS, **(params or {})}
    data = df.dropna(subset=[target])
    if len(data) < 10:
        raise ValueError("Need at least 10 rows with a non-missing target.")

    spec = FeatureSpec.fit(data[features], force_categorical=categorical)
    X = spec.transform(data)
    y_raw = data[target]

    class_names: list[str] = []
    if task == REGRESSION:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        keep = ~np.isnan(y)
        X, y = X[keep], y[keep]
        stratify = None
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        class_names = [str(c) for c in le.classes_]
        n_classes = len(class_names)
        if task == BINARY and n_classes != 2:
            raise ValueError(f"Binary task needs 2 classes, found {n_classes}.")
        if task == MULTICLASS and n_classes < 3:
            task = BINARY
        counts = np.bincount(y)
        stratify = y if counts.min() >= 3 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    X_fit, y_fit, X_val, y_val = X_train, y_train, None, None
    if early_stopping_rounds and val_size > 0:
        strat_tr = y_train if stratify is not None else None
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=seed, stratify=strat_tr
        )

    common = dict(
        **params,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=4,
        random_state=seed,
        n_jobs=-1,
        device="cuda" if use_gpu else "cpu",
    )
    if early_stopping_rounds and X_val is not None:
        common["early_stopping_rounds"] = early_stopping_rounds

    sample_weight = None
    if task == REGRESSION:
        eval_metric = "rmse"
        model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric=eval_metric, **common)
    elif task == BINARY:
        eval_metric = "logloss"
        if balance_classes:
            neg, pos = np.bincount(y_fit, minlength=2)
            common["scale_pos_weight"] = neg / max(pos, 1)
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric=eval_metric, **common)
    else:
        eval_metric = "mlogloss"
        if balance_classes:
            freq = np.bincount(y_fit) / len(y_fit)
            sample_weight = (1.0 / (freq * len(freq)))[y_fit]
        model = xgb.XGBClassifier(objective="multi:softprob", eval_metric=eval_metric, **common)

    eval_set = [(X_fit, y_fit)] + ([(X_val, y_val)] if X_val is not None else [])
    model.fit(X_fit, y_fit, sample_weight=sample_weight, eval_set=eval_set, verbose=False)

    raw_hist = model.evals_result()
    history = {
        ("train" if i == 0 else "validation"): v[eval_metric]
        for i, v in enumerate(raw_hist.values())
    }

    result = TrainResult(
        booster=model.get_booster(),
        model=model,
        task=task,
        target=target,
        spec=spec,
        class_names=class_names,
        best_iteration=getattr(model, "best_iteration", None) if "early_stopping_rounds" in common else None,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metrics={},
        history=history,
        eval_metric=eval_metric,
        params=params,
    )
    result.metrics = evaluate(result)
    return result


def evaluate(r: TrainResult) -> dict[str, float]:
    m: dict[str, float] = {}
    if r.task == REGRESSION:
        pred = r.test_pred
        m["RMSE"] = float(np.sqrt(mean_squared_error(r.y_test, pred)))
        m["MAE"] = float(mean_absolute_error(r.y_test, pred))
        m["R²"] = float(r2_score(r.y_test, pred))
    else:
        proba = r.test_pred
        pred = proba.argmax(axis=1)
        labels = list(range(len(r.class_names)))
        m["Accuracy"] = float(accuracy_score(r.y_test, pred))
        m["F1 (macro)"] = float(f1_score(r.y_test, pred, average="macro"))
        m["Log loss"] = float(log_loss(r.y_test, proba, labels=labels))
        try:
            if r.task == BINARY:
                m["ROC AUC"] = float(roc_auc_score(r.y_test, proba[:, 1]))
            else:
                m["ROC AUC (ovr)"] = float(
                    roc_auc_score(r.y_test, proba, multi_class="ovr", labels=labels)
                )
        except ValueError:
            pass  # a class absent from the test split
    m["Trees used"] = float(r.n_trees_used)
    return m


def global_importance(r: Predictor, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature (summed over classes for multiclass)."""
    values, _ = r.shap_values(X)
    imp = np.abs(values).mean(axis=0)
    if imp.ndim == 2:  # (k, f)
        imp = imp.sum(axis=0)
    gain = r.booster.get_score(importance_type="gain")
    return (
        pd.DataFrame(
            {
                "feature": r.feature_names,
                "mean_abs_shap": imp,
                "gain": [gain.get(f, 0.0) for f in r.feature_names],
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def shap_frame(r: Predictor, df: pd.DataFrame, transformed: bool = False) -> pd.DataFrame:
    """Predictions plus one SHAP column per feature (per class for multiclass)."""
    X = df if transformed else r.spec.transform(df)
    values, base = r.shap_values(X)
    out = r.predict_frame(X, transformed=True)
    if values.ndim == 2:
        out["shap_base_value"] = base
        for j, f in enumerate(r.feature_names):
            out[f"shap_{f}"] = values[:, j]
    else:
        for k, cls in enumerate(r.class_names):
            out[f"shap_base_value[{cls}]"] = base[:, k]
            for j, f in enumerate(r.feature_names):
                out[f"shap_{f}[{cls}]"] = values[:, k, j]
    return out


# --------------------------------------------------------------------------- #
# Model export / import
# --------------------------------------------------------------------------- #
META_ATTR = "xgb_shap_app"  # booster attribute holding this app's metadata
META_VERSION = 1


def model_json_bytes(r: Predictor) -> bytes:
    """XGBoost JSON model with this app's metadata embedded as a booster attribute.

    The file stays a standard XGBoost model: ``xgb.Booster(model_file=...)``
    loads it anywhere and ignores the extra attribute.
    """
    meta = {
        "version": META_VERSION,
        "task": r.task,
        "target": r.target,
        "class_names": r.class_names,
        "best_iteration": r.best_iteration,
        "spec": {
            "columns": r.spec.columns,
            "categories": r.spec.categories,
            "datetimes": r.spec.datetimes,
        },
    }
    if isinstance(r, TrainResult):
        meta["metrics"] = r.metrics
        meta["params"] = r.params
    booster = r.booster.copy()
    booster.set_attr(**{META_ATTR: json.dumps(meta, default=str)})
    return bytes(booster.save_raw("json"))


def load_model(data: bytes) -> Predictor:
    """Restore a Predictor from an XGBoost model file (.json or .ubj).

    Files exported by this app restore exactly. Other XGBoost models are
    reconstructed from the booster itself; anything that cannot be recovered
    is listed in ``Predictor.notes``.
    """
    booster = xgb.Booster()
    try:
        booster.load_model(bytearray(data))
    except xgb.core.XGBoostError as e:
        raise ValueError(f"Not a valid XGBoost model file: {e}") from e

    raw_meta = booster.attr(META_ATTR)
    if raw_meta:
        meta = json.loads(raw_meta)
        sp = meta["spec"]
        return Predictor(
            booster=booster,
            task=meta["task"],
            target=meta["target"],
            spec=FeatureSpec(columns=sp["columns"], categories=sp["categories"], datetimes=sp["datetimes"]),
            class_names=meta["class_names"],
            best_iteration=meta["best_iteration"],
        )
    return _predictor_from_foreign_booster(booster)


def _predictor_from_foreign_booster(booster: xgb.Booster) -> Predictor:
    notes: list[str] = []
    config = json.loads(booster.save_config())["learner"]
    objective = config["objective"]["name"]
    n_classes = int(config["learner_model_param"].get("num_class", "0") or 0)
    if objective.startswith("binary:"):
        task, class_names = BINARY, ["0", "1"]
    elif objective.startswith("multi:"):
        task, class_names = MULTICLASS, [str(i) for i in range(n_classes)]
    else:
        task, class_names = REGRESSION, []
        if not objective.startswith("reg:"):
            notes.append(f"Objective '{objective}' is treated as regression; predictions are the model's output.")
    if class_names:
        notes.append("Class labels are not stored in this model, so classes are shown as indices "
                     f"({', '.join(class_names)}).")

    names = booster.feature_names
    if names is None:
        names = [f"f{i}" for i in range(booster.num_features())]
        notes.append("Feature names are not stored in this model; columns must be named "
                     f"{names[0]} … {names[-1]} in the order used for training.")
    types = booster.feature_types or [None] * len(names)

    stored = _stored_categories(booster, len(names))
    categories: dict[str, list[Any] | None] = {}
    for i, (name, ftype) in enumerate(zip(names, types)):
        if ftype == "c":
            categories[name] = stored[i]
            if stored[i] is None:
                notes.append(f"Category list for '{name}' is not stored (model saved by XGBoost < 3.1); "
                             "categories are taken from the scoring data.")
    notes.append("Date columns can't be detected from a plain XGBoost model; any date features must be "
                 "supplied as numbers in the same units used for training.")

    best = booster.attr("best_iteration")
    return Predictor(
        booster=booster,
        task=task,
        target=booster.attr("target") or "target",
        spec=FeatureSpec(columns=list(names), categories=categories),
        class_names=class_names,
        best_iteration=int(best) if best is not None else None,
        notes=notes,
    )


def _stored_categories(booster: xgb.Booster, n_features: int) -> list[list[Any] | None]:
    """Decode the per-feature category lists XGBoost >= 3.1 saves with the model."""
    model = json.loads(bytes(booster.save_raw("json")))
    cats = model["learner"]["gradient_booster"]["model"].get("cats") or {}
    out: list[list[Any] | None] = [None] * n_features
    for i, enc in enumerate(cats.get("enc", [])[:n_features]):
        values = enc.get("values", [])
        if not values:
            continue
        offsets = enc.get("offsets")
        if offsets:  # strings: UTF-8 bytes split by offsets
            buf = bytes(values)
            out[i] = [buf[a:b].decode("utf-8") for a, b in zip(offsets[:-1], offsets[1:])]
        else:  # numeric categories
            out[i] = list(values)
    return out


EXCEL_MAX_ROWS = 1_048_575  # data rows per sheet (1,048,576 minus header)
EXCEL_MAX_COLS = 16_384


def to_excel_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    """Write frames to .xlsx, splitting any frame over Excel's row limit into
    numbered sheets (name_1, name_2, ...)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, frame in frames.items():
            if frame.shape[1] > EXCEL_MAX_COLS:
                raise ValueError(
                    f"'{name}' has {frame.shape[1]:,} columns; Excel allows {EXCEL_MAX_COLS:,}. Use CSV."
                )
            chunks = range(0, max(len(frame), 1), EXCEL_MAX_ROWS)
            for i, start in enumerate(chunks, 1):
                sheet = name[:31] if len(chunks) == 1 else f"{name[:27]}_{i}"
                frame.iloc[start : start + EXCEL_MAX_ROWS].to_excel(xw, sheet_name=sheet, index=False)
    return buf.getvalue()
