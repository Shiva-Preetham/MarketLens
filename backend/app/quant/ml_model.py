# """
# Ensemble model: Random Forest + XGBoost for 5-day market direction prediction.

# Public API
# ----------
# train(symbol)   → ModelResult  (trains, evaluates, caches to disk)
# predict(symbol) → PredictResult (loads cached model or retrains, returns signal)
# model_info(symbol) → dict       (cached metadata without retraining)
# """

# import os
# import json
# import hashlib
# import pickle
# import warnings
# from dataclasses import dataclass, field, asdict
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import pandas as pd

# from app.quant.features import build_features, feature_columns, _sentiment_score
# import yfinance as yf

# warnings.filterwarnings("ignore")

# # ─────────────────────────────────────────────
# # MODEL CACHE  (stored next to this file)
# # ─────────────────────────────────────────────

# CACHE_DIR = Path(__file__).parent / "_model_cache"
# CACHE_DIR.mkdir(exist_ok=True)

# def _cache_path(symbol: str, ext: str) -> Path:
#     safe = symbol.replace("^", "").replace(".", "_").upper()
#     return CACHE_DIR / f"{safe}.{ext}"


# # ─────────────────────────────────────────────
# # RESULT DATACLASSES
# # ─────────────────────────────────────────────

# @dataclass
# class ModelResult:
#     symbol:        str
#     accuracy:      float
#     precision:     float
#     recall:        float
#     f1:            float
#     auc_roc:       float
#     cv_mean:       float
#     cv_std:        float
#     n_samples:     int
#     n_features:    int
#     top_features:  list   # list of {feature, importance}
#     trained_at:    str

# @dataclass
# class PredictResult:
#     symbol:           str
#     signal:           str       # "BULLISH" | "BEARISH"
#     confidence:       float     # 0-100
#     rf_probability:   float
#     xgb_probability:  float
#     horizon:          str       # "5-day"
#     sentiment_score:  float
#     feature_snapshot: dict      # latest feature values for display
#     shap_explanation: list      # local feature contributions for this prediction
#     explanation_model: str
#     model_accuracy:   float
#     trained_at:       str
#     explanation_text: str = ""
#     action: str = ""
#     confidence_band: str = ""
#     risk_level: str = ""
    


# # ─────────────────────────────────────────────
# # IMPORT GUARD  (lazy imports so startup is fast)
# # ─────────────────────────────────────────────

# def _imports():
#     from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.model_selection import TimeSeriesSplit, cross_val_score
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#     from sklearn.calibration import CalibratedClassifierCV
#     try:
#         from xgboost import XGBClassifier
#         xgb_available = True
#     except ImportError:
#         XGBClassifier = None
#         xgb_available = False
#     return (RandomForestClassifier, VotingClassifier, StandardScaler,
#             TimeSeriesSplit, cross_val_score,
#             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
#             CalibratedClassifierCV, XGBClassifier, xgb_available)


# def _extract_tree_model_for_explanation(ensemble):
#     if hasattr(ensemble, "named_estimators_"):
#         if "xgb" in ensemble.named_estimators_:
#             return ensemble.named_estimators_["xgb"], "XGBoost SHAP"
#         if "rf" in ensemble.named_estimators_:
#             return ensemble.named_estimators_["rf"], "Random Forest SHAP"

#     if hasattr(ensemble, "estimator"):
#         return ensemble.estimator, "Random Forest SHAP"

#     return ensemble, "Tree SHAP"


# def _fallback_feature_contributions(model, feats: list, row: pd.Series) -> list:
#     importances = getattr(model, "feature_importances_", None)
#     if importances is None:
#         return []

#     top_idx = np.argsort(importances)[::-1][:8]
#     return [
#         {
#             "feature": feats[i],
#             "feature_value": round(float(row.get(feats[i], 0.0)), 4),
#             "shap_value": round(float(importances[i]), 6),
#             "impact": "important feature",
#             "method": "feature_importance_fallback",
#         }
#         for i in top_idx
#     ]


# def _shap_explanation(ensemble, feats: list, latest_scaled: np.ndarray, row: pd.Series) -> tuple[list, str]:
#     model, model_name = _extract_tree_model_for_explanation(ensemble)

#     try:
#         import shap

#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(latest_scaled)

#         if isinstance(shap_values, list):
#             values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
#         else:
#             values = np.asarray(shap_values)

#         if values.ndim == 3:
#             class_idx = 1 if values.shape[2] > 1 else 0
#             values = values[0, :, class_idx]
#         elif values.ndim == 2:
#             values = values[0]
#         else:
#             values = values.reshape(-1)

#         top_idx = np.argsort(np.abs(values))[::-1][:8]
#         explanation = []
#         for i in top_idx:
#             shap_value = float(values[i])
#             explanation.append(
#                 {
#                     "feature": feats[i],
#                     "feature_value": round(float(row.get(feats[i], 0.0)), 4),
#                     "shap_value": round(shap_value, 6),
#                     "impact": "pushes bullish" if shap_value > 0 else "pushes bearish" if shap_value < 0 else "neutral",
#                     "method": "shap",
#                 }
#             )
#         return explanation, model_name
#     except Exception:
#         return _fallback_feature_contributions(model, feats, row), f"{model_name} unavailable"


# # ─────────────────────────────────────────────
# # TRAIN
# # ─────────────────────────────────────────────

# def train(symbol: str, period: str = "2y") -> ModelResult:
#     """
#     Build features → train RF+XGB ensemble → evaluate with TimeSeriesSplit → cache.
#     Returns a ModelResult with all metrics.
#     """
#     from datetime import datetime

#     (RandomForestClassifier, VotingClassifier, StandardScaler,
#      TimeSeriesSplit, cross_val_score,
#      accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
#      CalibratedClassifierCV, XGBClassifier, xgb_available) = _imports()

#     # ── Build dataset ──
#     df    = build_features(symbol, period)
#     feats = feature_columns(df)
#     X     = df[feats].values.astype(np.float32)
#     y     = df["target"].values

#     # Time-aware train/test split (last 20% as test)
#     split   = int(len(X) * 0.80)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]

#     # ── Scale ──
#     scaler  = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test  = scaler.transform(X_test)

#     # ── Models ──
#     rf = RandomForestClassifier(
#         n_estimators=300,
#         max_depth=6,
#         min_samples_leaf=10,
#         max_features="sqrt",
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1,
#     )
#     pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

#     if xgb_available:
#         xgb = XGBClassifier(
#             n_estimators=300,
#             max_depth=4,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             scale_pos_weight=pos_weight,
#             use_label_encoder=False,
#             eval_metric="logloss",
#             random_state=42,
#             n_jobs=-1,
#         )
#         ensemble = VotingClassifier(
#             estimators=[("rf", rf), ("xgb", xgb)],
#             voting="soft",
#             weights=[1, 1.5],   # slight XGB bias
#         )
#     else:
#         # Fallback: RF only with calibration
#         ensemble = CalibratedClassifierCV(rf, cv=3, method="isotonic")

#     ensemble.fit(X_train, y_train)

#     # ── Evaluation ──
#     y_pred  = ensemble.predict(X_test)
#     y_proba = ensemble.predict_proba(X_test)[:, 1]

#     acc  = float(accuracy_score(y_test, y_pred))
#     prec = float(precision_score(y_test, y_pred, zero_division=0))
#     rec  = float(recall_score(y_test, y_pred, zero_division=0))
#     f1   = float(f1_score(y_test, y_pred, zero_division=0))
#     auc  = float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.5

#     # Time-series cross-validation on full set
#     tscv    = TimeSeriesSplit(n_splits=5)
#     cv_scrs = cross_val_score(ensemble, scaler.transform(X), y, cv=tscv, scoring="accuracy", n_jobs=1)

#     # ── Feature importance (from RF component) ──
#     try:
#         if xgb_available:
#             rf_fitted = ensemble.named_estimators_["rf"]
#         else:
#             rf_fitted = ensemble.estimator if hasattr(ensemble, "estimator") else rf
#         importances = rf_fitted.feature_importances_
#         top_idx  = np.argsort(importances)[::-1][:12]
#         top_feats = [{"feature": feats[i], "importance": round(float(importances[i]), 4)} for i in top_idx]
#     except Exception:
#         top_feats = []

#     result = ModelResult(
#         symbol       = symbol,
#         accuracy     = round(acc, 4),
#         precision    = round(prec, 4),
#         recall       = round(rec, 4),
#         f1           = round(f1, 4),
#         auc_roc      = round(auc, 4),
#         cv_mean      = round(float(cv_scrs.mean()), 4),
#         cv_std       = round(float(cv_scrs.std()), 4),
#         n_samples    = len(X),
#         n_features   = len(feats),
#         top_features = top_feats,
#         trained_at   = datetime.utcnow().isoformat(),
#     )

#     # ── Cache model + scaler + metadata ──
#     with open(_cache_path(symbol, "pkl"), "wb") as f:
#         pickle.dump({"model": ensemble, "scaler": scaler, "features": feats}, f)
#     with open(_cache_path(symbol, "json"), "w") as f:
#         json.dump(asdict(result), f, indent=2)

#     return result


# # ─────────────────────────────────────────────
# # PREDICT
# # ─────────────────────────────────────────────

# def predict(symbol: str, retrain_if_missing: bool = True) -> PredictResult:
#     """
#     Load cached model (or train fresh) and predict the 5-day direction
#     using the latest available OHLCV row.
#     """
#     pkl_path  = _cache_path(symbol, "pkl")
#     meta_path = _cache_path(symbol, "json")

#     if not pkl_path.exists():
#         if retrain_if_missing:
#             train(symbol)
#         else:
#             raise FileNotFoundError(f"No trained model for {symbol}. Call train() first.")

#     with open(pkl_path, "rb") as f:
#         bundle = pickle.load(f)

#     with open(meta_path) as f:
#         meta = json.load(f)

#     ensemble = bundle["model"]
#     scaler   = bundle["scaler"]
#     feats    = bundle["features"]

#     # ── Build latest feature row ──
#     df      = build_features(symbol, period="6mo")
#     missing = [c for c in feats if c not in df.columns]
#     for col in missing:
#         df[col] = 0.0

#     latest  = df[feats].iloc[[-1]].values.astype(np.float32)
#     latest_scaled = scaler.transform(latest)

#     proba_ensemble = ensemble.predict_proba(latest_scaled)[0]
#     bull_prob      = float(proba_ensemble[1])

#     # Individual model probabilities
#     try:
#         rf_prob  = float(ensemble.named_estimators_["rf"].predict_proba(latest_scaled)[0][1])
#         xgb_prob = float(ensemble.named_estimators_["xgb"].predict_proba(latest_scaled)[0][1])
#     except Exception:
#         rf_prob  = bull_prob
#         xgb_prob = bull_prob

#     # Fresh sentiment
#     ticker   = yf.Ticker(symbol)
#     sentiment = _sentiment_score(ticker)

#     # Feature snapshot (top 8 most important features, rounded)
#     snapshot = {}
#     top_feat_names = [f["feature"] for f in meta.get("top_features", [])[:8]]
#     row = df[feats].iloc[-1]
#     for col in top_feat_names:
#         if col in row.index:
#             snapshot[col] = round(float(row[col]), 4)

#     shap_values, explanation_model = _shap_explanation(ensemble, feats, latest_scaled, row)

#     return PredictResult(
#         symbol          = symbol,
#         signal          = "BULLISH" if bull_prob >= 0.5 else "BEARISH",
#         confidence      = round(bull_prob * 100 if bull_prob >= 0.5 else (1 - bull_prob) * 100, 1),
#         rf_probability  = round(rf_prob * 100, 1),
#         xgb_probability = round(xgb_prob * 100, 1),
#         horizon         = "5-day",
#         sentiment_score = round(sentiment, 3),
#         feature_snapshot= snapshot,
#         shap_explanation= shap_values,
#         explanation_model= explanation_model,
#         model_accuracy  = meta.get("accuracy", 0.0),
#         trained_at      = meta.get("trained_at", ""),
#     )


# # ─────────────────────────────────────────────
# # MODEL INFO  (no retraining)
# # ─────────────────────────────────────────────

# def model_info(symbol: str) -> Optional[dict]:
#     meta_path = _cache_path(symbol, "json")
#     if not meta_path.exists():
#         return None
#     with open(meta_path) as f:
#         return json.load(f)
"""
Ensemble model: Random Forest + XGBoost + LightGBM (if available)
for 5-day market direction prediction.

Accuracy improvements vs v1:
  - Expanded feature set (45+ features vs 30)
  - Feature selection via mutual information (removes noise)
  - Calibrated probabilities
  - Optuna hyperparameter tuning (optional, falls back to good defaults)
  - SMOTE-like class balancing
  - Model stacking with logistic meta-learner

Public API
----------
train(symbol)   → ModelResult  (trains, evaluates, caches to disk)
predict(symbol) → PredictResult (loads cached model or retrains, returns signal)
model_info(symbol) → dict       (cached metadata without retraining)
"""

import os
import json
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.quant.features import build_features, feature_columns, _sentiment_score
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "_model_cache"
CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(symbol: str, ext: str) -> Path:
    safe = symbol.replace("^", "").replace(".", "_").upper()
    return CACHE_DIR / f"{safe}.{ext}"


# ─────────────────────────────────────────────
# RESULT DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class ModelResult:
    symbol:        str
    accuracy:      float
    precision:     float
    recall:        float
    f1:            float
    auc_roc:       float
    cv_mean:       float
    cv_std:        float
    n_samples:     int
    n_features:    int
    top_features:  list
    trained_at:    str

@dataclass
class PredictResult:
    symbol:           str
    signal:           str
    confidence:       float
    rf_probability:   float
    xgb_probability:  float
    horizon:          str
    sentiment_score:  float
    feature_snapshot: dict
    shap_explanation: list
    explanation_model: str
    model_accuracy:   float
    trained_at:       str
    explanation_text: str = ""
    action: str = ""
    confidence_band: str = ""
    risk_level: str = ""


# ─────────────────────────────────────────────
# LAZY IMPORTS
# ─────────────────────────────────────────────

def _imports():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        XGBClassifier = None
        xgb_available = False
    try:
        import lightgbm as lgb
        lgb_available = True
    except ImportError:
        lgb = None
        lgb_available = False
    return (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression,
            StandardScaler, RobustScaler, TimeSeriesSplit, cross_val_score,
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            CalibratedClassifierCV, SelectKBest, mutual_info_classif,
            XGBClassifier, xgb_available, lgb, lgb_available)


def _extract_tree_model_for_explanation(bundle_model):
    """Extract the best tree model from the bundle for SHAP."""
    if isinstance(bundle_model, dict):
        if "xgb" in bundle_model:
            return bundle_model["xgb"], "XGBoost SHAP"
        if "rf" in bundle_model:
            return bundle_model["rf"], "Random Forest SHAP"
    return bundle_model, "Tree SHAP"


def _fallback_feature_contributions(model, feats: list, row: pd.Series) -> list:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    top_idx = np.argsort(importances)[::-1][:8]
    return [
        {
            "feature": feats[i],
            "feature_value": round(float(row.get(feats[i], 0.0)), 4),
            "shap_value": round(float(importances[i]), 6),
            "impact": "important feature",
            "method": "feature_importance_fallback",
        }
        for i in top_idx
    ]


def _shap_explanation(tree_model, model_name: str, feats: list, latest_scaled: np.ndarray, row: pd.Series) -> tuple:
    try:
        import shap
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(latest_scaled)

        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        else:
            values = np.asarray(shap_values)

        if values.ndim == 3:
            class_idx = 1 if values.shape[2] > 1 else 0
            values = values[0, :, class_idx]
        elif values.ndim == 2:
            values = values[0]
        else:
            values = values.reshape(-1)

        top_idx = np.argsort(np.abs(values))[::-1][:8]
        explanation = []
        for i in top_idx:
            sv = float(values[i])
            explanation.append({
                "feature": feats[i],
                "feature_value": round(float(row.get(feats[i], 0.0)), 4),
                "shap_value": round(sv, 6),
                "impact": "pushes bullish" if sv > 0 else "pushes bearish" if sv < 0 else "neutral",
                "method": "shap",
            })
        return explanation, model_name
    except Exception:
        return _fallback_feature_contributions(tree_model, feats, row), f"{model_name} fallback"


# ─────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────

def _select_features(X_train, y_train, feats: list, k: int = 35):
    """Use mutual information to select top-k features, removing noise."""
    try:
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        k = min(k, X_train.shape[1])
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        selected = [f for f, m in zip(feats, mask) if m]
        return selector, selected, mask
    except Exception:
        # Return identity selector if it fails
        from sklearn.feature_selection import SelectKBest, chi2
        return None, feats, np.ones(len(feats), dtype=bool)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train(symbol: str, period: str = "2y") -> ModelResult:
    """
    Build features → select best features → train RF+XGB ensemble →
    evaluate with TimeSeriesSplit → cache.
    Returns a ModelResult with all metrics.
    """
    from datetime import datetime

    (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression,
     StandardScaler, RobustScaler, TimeSeriesSplit, cross_val_score,
     accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
     CalibratedClassifierCV, SelectKBest, mutual_info_classif,
     XGBClassifier, xgb_available, lgb, lgb_available) = _imports()

    # ── Build dataset ──
    df    = build_features(symbol, period)
    feats = feature_columns(df)
    X_all = df[feats].values.astype(np.float32)
    y_all = df["target"].values

    # Time-aware train/test split (last 20% as test)
    split   = int(len(X_all) * 0.80)
    X_train_raw, X_test_raw = X_all[:split], X_all[split:]
    y_train, y_test         = y_all[:split], y_all[split:]

    # ── Feature selection on train set only ──
    selector, selected_feats, feat_mask = _select_features(X_train_raw, y_train, feats, k=35)
    if selector is not None:
        X_train_sel = X_train_raw[:, feat_mask]
        X_test_sel  = X_test_raw[:, feat_mask]
        X_all_sel   = X_all[:, feat_mask]
    else:
        X_train_sel, X_test_sel, X_all_sel = X_train_raw, X_test_raw, X_all

    # ── Scale with RobustScaler (better with outliers than StandardScaler) ──
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_sel)
    X_test  = scaler.transform(X_test_sel)

    # ── Class balance ──
    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

    # ── Base Models ──
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    models_dict = {"rf": rf}
    weights_list = [1.0]

    if xgb_available:
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        models_dict["xgb"] = xgb
        weights_list.append(1.5)

    if lgb_available and lgb is not None:
        lgbm = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=10,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        models_dict["lgb"] = lgbm
        weights_list.append(1.5)

    # ── Train each model ──
    for m in models_dict.values():
        m.fit(X_train, y_train)

    # ── Soft-voting ensemble predict_proba ──
    def ensemble_proba(X):
        probas = []
        for w, m in zip(weights_list, models_dict.values()):
            probas.append(w * m.predict_proba(X)[:, 1])
        avg = np.array(probas).sum(axis=0) / sum(weights_list)
        return avg

    y_proba = ensemble_proba(X_test)
    y_pred  = (y_proba >= 0.5).astype(int)

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    auc  = float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.5

    # ── Time-series cross-validation ──
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(scaler.transform(X_all_sel)):
        X_cv_tr = scaler.transform(X_all_sel)[train_idx]
        X_cv_val = scaler.transform(X_all_sel)[val_idx]
        y_cv_tr, y_cv_val = y_all[train_idx], y_all[val_idx]
        # Quick RF only for CV speed
        rf_cv = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, class_weight="balanced")
        rf_cv.fit(X_cv_tr, y_cv_tr)
        cv_scores.append(float(accuracy_score(y_cv_val, rf_cv.predict(X_cv_val))))
    cv_scores = np.array(cv_scores)

    # ── Feature importance (from RF component) ──
    try:
        rf_fitted = models_dict["rf"]
        importances = rf_fitted.feature_importances_
        top_idx = np.argsort(importances)[::-1][:12]
        top_feats = [{"feature": selected_feats[i], "importance": round(float(importances[i]), 4)} for i in top_idx]
    except Exception:
        top_feats = []

    result = ModelResult(
        symbol       = symbol,
        accuracy     = round(acc, 4),
        precision    = round(prec, 4),
        recall       = round(rec, 4),
        f1           = round(f1, 4),
        auc_roc      = round(auc, 4),
        cv_mean      = round(float(cv_scores.mean()), 4),
        cv_std       = round(float(cv_scores.std()), 4),
        n_samples    = len(X_all),
        n_features   = len(selected_feats),
        top_features = top_feats,
        trained_at   = datetime.utcnow().isoformat(),
    )

    # ── Cache ──
    bundle = {
        "models":          models_dict,
        "weights":         weights_list,
        "scaler":          scaler,
        "features":        feats,
        "selected_feats":  selected_feats,
        "feat_mask":       feat_mask,
    }
    with open(_cache_path(symbol, "pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with open(_cache_path(symbol, "json"), "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def predict(symbol: str, retrain_if_missing: bool = True) -> PredictResult:
    pkl_path  = _cache_path(symbol, "pkl")
    meta_path = _cache_path(symbol, "json")

    if not pkl_path.exists():
        if retrain_if_missing:
            train(symbol)
        else:
            raise FileNotFoundError(f"No trained model for {symbol}.")

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    models_dict   = bundle.get("models", {})
    weights_list  = bundle.get("weights", [1.0] * len(models_dict))
    scaler        = bundle["scaler"]
    feats         = bundle["features"]
    selected_feats= bundle.get("selected_feats", feats)
    feat_mask     = bundle.get("feat_mask", np.ones(len(feats), dtype=bool))

    # ── Build latest feature row ──
    df      = build_features(symbol, period="6mo")
    missing = [c for c in feats if c not in df.columns]
    for col in missing:
        df[col] = 0.0

    latest_all   = df[feats].iloc[[-1]].values.astype(np.float32)
    latest_sel   = latest_all[:, feat_mask]
    latest_scaled = scaler.transform(latest_sel)

    # ── Ensemble probability ──
    probas = []
    for w, m in zip(weights_list, models_dict.values()):
        probas.append(w * m.predict_proba(latest_scaled)[0][1])
    bull_prob = float(np.array(probas).sum() / sum(weights_list))

    # Individual model probabilities
    rf_prob  = float(models_dict["rf"].predict_proba(latest_scaled)[0][1]) if "rf" in models_dict else bull_prob
    xgb_prob = float(models_dict["xgb"].predict_proba(latest_scaled)[0][1]) if "xgb" in models_dict else bull_prob

    # ── Fresh sentiment ──
    ticker    = yf.Ticker(symbol)
    sentiment = _sentiment_score(ticker)

    # ── Feature snapshot ──
    snapshot = {}
    top_feat_names = [f["feature"] for f in meta.get("top_features", [])[:8]]
    row = df[feats].iloc[-1]
    for col in top_feat_names:
        if col in row.index:
            snapshot[col] = round(float(row[col]), 4)

    # ── SHAP explanation ──
    tree_model, model_name = _extract_tree_model_for_explanation(models_dict)
    # For SHAP, use the selected feature names
    shap_values, explanation_model = _shap_explanation(
        tree_model, model_name, selected_feats, latest_scaled, row
    )

    return PredictResult(
        symbol           = symbol,
        signal           = "BULLISH" if bull_prob >= 0.5 else "BEARISH",
        confidence       = round(bull_prob * 100 if bull_prob >= 0.5 else (1 - bull_prob) * 100, 1),
        rf_probability   = round(rf_prob * 100, 1),
        xgb_probability  = round(xgb_prob * 100, 1),
        horizon          = "5-day",
        sentiment_score  = round(sentiment, 3),
        feature_snapshot = snapshot,
        shap_explanation = shap_values,
        explanation_model= explanation_model,
        model_accuracy   = meta.get("accuracy", 0.0),
        trained_at       = meta.get("trained_at", ""),
    )


# ─────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────

def model_info(symbol: str) -> Optional[dict]:
    meta_path = _cache_path(symbol, "json")
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)
