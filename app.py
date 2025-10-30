import io, base64, joblib
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="LeadScorer v2 — Advanced", page_icon="📈", layout="wide")
st.title("📈 LeadScorer v2 — Advanced Lead Scoring (browser-only)")
st.caption("Train, tune, calibrate, pick a business-optimal threshold, and score new leads. All in your browser.")

# ---------------- Sidebar: data sources ----------------
with st.sidebar:
    st.header("Data sources")
    st.write("1) Upload CSV **or** 2) Paste a Google Sheets CSV URL")
    st.code("https://docs.google.com/spreadsheets/d/.../pub?output=csv")
    gs_url = st.text_input("Google Sheets CSV URL (optional)")
    st.markdown("---")
    st.header("Model & tuning")
    model_choice = st.selectbox("Model", ["Gradient Boosting", "Logistic Regression"])
    do_tune = st.checkbox("Hyperparameter tuning (RandomizedSearchCV)", value=True)
    calibrate = st.checkbox("Calibrate probabilities (recommended)", value=True)
    st.markdown("---")
    st.header("Business costs (for threshold)")
    cost_fp = st.number_input("Cost of a false positive (e.g., banker time $)", 0.0, 1e6, 10.0, step=1.0)
    cost_fn = st.number_input("Cost of a false negative (missed opp $)", 0.0, 1e6, 50.0, step=1.0)

st.subheader("1) Upload training data")
train_file = st.file_uploader("Training CSV (must include a binary label)", type=["csv"])

DEMO = st.checkbox("Use demo training data", value=(train_file is None and not gs_url))
if DEMO:
    df_train = pd.read_csv("data/leads_training_sample.csv")
elif gs_url:
    df_train = pd.read_csv(gs_url)
elif train_file:
    df_train = pd.read_csv(train_file)
else:
    st.info("Upload a file or paste a Google Sheets CSV URL, or toggle demo.")
    st.stop()

st.write("**Training preview:**")
st.dataframe(df_train.head(10), use_container_width=True)

# Select label & features
all_cols = list(df_train.columns)
target_col = st.selectbox("Label column (0/1)", options=all_cols, index=all_cols.index("closed_loan") if "closed_loan" in all_cols else 0)
feature_cols_default = [c for c in all_cols if c != target_col and c not in ("lead_id",)]
feature_cols = st.multiselect("Feature columns", options=[c for c in all_cols if c != target_col], default=feature_cols_default)
seg_cols = st.multiselect("Segment-by columns (for sliced metrics)", options=feature_cols, default=[c for c in ("channel","device") if c in feature_cols])

if not feature_cols:
    st.warning("Pick at least one feature.")
    st.stop()

df_model = df_train.dropna(subset=[target_col]).copy()
y = df_model[target_col].astype(int)
X = df_model[feature_cols]

# Identify types
cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Preprocess
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False))  # robust for sparse combos
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

# Base estimators
if model_choice == "Gradient Boosting":
    base_est = GradientBoostingClassifier(random_state=42)
    param_dist = {
        "clf__n_estimators": [80, 120, 160, 200],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": [0.03, 0.05, 0.07, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
    }
else:
    base_est = LogisticRegression(max_iter=200, n_jobs=None, solver="liblinear")
    param_dist = {
        "clf__C": np.logspace(-2, 2, 10),
        "clf__penalty": ["l1","l2"]
    }

pipe = Pipeline(steps=[("prep", preprocess), ("clf", base_est)])

# Train/test split
st.subheader("2) Train & evaluate")
cA, cB, cC = st.columns([1,1,2])
with cA:
    test_size = st.slider("Holdout %", 0.1, 0.4, 0.25, 0.05)
with cB:
    cv_folds = st.slider("CV folds (tuning)", 3, 7, 5, 1)
train_btn = st.button("🚀 Train / Tune")

if train_btn:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        best_pipe = pipe
        if do_tune:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=15,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            st.info(f"Best params: {search.best_params_}")

        if calibrate:
            # Calibrate on a split from training to avoid peeking at test
            X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
            best_pipe.fit(X_tr, y_tr)
            calibrated = CalibratedClassifierCV(best_pipe.named_steps["clf"], cv="prefit", method="isotonic")
            # wrap: we need the same preprocessor + calibrated classifier
            cal_pipe = Pipeline(steps=[
                ("prep", best_pipe.named_steps["prep"]),
                ("clf", calibrated)
            ])
            cal_pipe.fit(X_cal, y_cal)
            final_pipe = cal_pipe
        else:
            best_pipe.fit(X_train, y_train)
            final_pipe = best_pipe

        # Evaluate
        prob_test = final_pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob_test)
        ap = average_precision_score(y_test, prob_test)
        st.success(f"AUC: **{auc:.3f}**  |  Average Precision: **{ap:.3f}**")

        # Curves
        fpr, tpr, thr = roc_curve(y_test, prob_test)
        roc_fig = px.line(x=fpr, y=tpr, labels={"x":"False Positive Rate", "y":"True Positive Rate"}, title="ROC Curve")
        st.plotly_chart(roc_fig, use_container_width=True)

        pr_p, pr_r, pr_t = precision_recall_curve(y_test, prob_test)
        pr_fig = px.line(x=pr_r, y=pr_p, labels={"x":"Recall", "y":"Precision"}, title="Precision-Recall Curve")
        st.plotly_chart(pr_fig, use_container_width=True)

        # Threshold + business cost optimal
        st.subheader("3) Threshold & business objective")
        thr_default = 0.5
        thr_slider = st.slider("Score threshold", 0.01, 0.99, float(thr_default), 0.01)
        y_pred = (prob_test >= thr_slider).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        biz_cost = fp * cost_fp + fn * cost_fn
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("TP", tp); c2.metric("FP", fp); c3.metric("FN", fn); c4.metric("TN", tn); c5.metric("Biz cost", f"${biz_cost:,.0f}")

        # Sweep thresholds to find cost-minimizing cutoff
        steps = np.linspace(0.01, 0.99, 99)
        sweep = []
        for t in steps:
            yp = (prob_test >= t).astype(int)
            tn_, fp_, fn_, tp_ = confusion_matrix(y_test, yp).ravel()
            sweep.append({"threshold": t, "biz_cost": fp_*cost_fp + fn_*cost_fn})
        sweep_df = pd.DataFrame(sweep)
        best_row = sweep_df.loc[sweep_df["biz_cost"].idxmin()]
        st.caption(f"Min-cost threshold ≈ **{best_row.threshold:.2f}**  (cost ${best_row.biz_cost:,.0f})")
        cost_fig = px.line(sweep_df, x="threshold", y="biz_cost", title="Business Cost vs Threshold")
        cost_fig.add_vline(x=float(best_row.threshold), line_dash="dash", line_color="red")
        st.plotly_chart(cost_fig, use_container_width=True)

        # Segment-wise metrics
        if seg_cols:
            st.subheader("4) Segment metrics")
            seg_show = seg_cols[:2]  # keep table compact
            seg_df = X_test.copy()
            seg_df["y_true"] = y_test.values
            seg_df["score"] = prob_test
            seg_df["y_pred"] = y_pred
            group = seg_df.groupby(seg_show, dropna=False)
            rows = []
            for keys, g in group:
                if len(g) >= 10:
                    auc_g = roc_auc_score(g["y_true"], g["score"]) if len(g["y_true"].unique())>1 else np.nan
                    tn_, fp_, fn_, tp_ = confusion_matrix(g["y_true"], g["y_pred"]).ravel()
                    rows.append({
                        **({seg_show[0]: keys[0]} if len(seg_show)>=1 else {}),
                        **({seg_show[1]: keys[1]} if len(seg_show)>=2 else {}),
                        "n": len(g), "AUC": round(auc_g,3) if not np.isnan(auc_g) else None,
                        "TP": tp_, "FP": fp_, "FN": fn_, "TN": tn_
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("AUC", ascending=False), use_container_width=True)

        # Importance (permutation for consistency across models)
        st.subheader("5) Feature importance (permutation)")
        # Use a small sample for speed
        samp = min(2000, len(X_test))
        idx = np.random.choice(np.arange(len(X_test)), size=samp, replace=False)
        X_samp = X_test.iloc[idx]
        y_samp = y_test.iloc[idx]
        r = permutation_importance(final_pipe, X_samp, y_samp, n_repeats=5, random_state=42, scoring="roc_auc")
        # Get feature names after OHE:
        oh = final_pipe.named_steps["prep"].transformers_[1][1].named_steps["oh"] if cat_cols else None
        cat_names = list(oh.get_feature_names_out(cat_cols)) if oh else []
        feature_names = num_cols + cat_names
        imp_df = pd.DataFrame({"feature": feature_names, "importance": r.importances_mean}).sort_values("importance", ascending=False)
        st.dataframe(imp_df.head(30), use_container_width=True)
        st.plotly_chart(px.bar(imp_df.head(20), x="importance", y="feature", orientation="h", title="Top Features"), use_container_width=True)

        # Save model for later download
        st.subheader("6) Save / load model")
        bytes_io = io.BytesIO()
        joblib.dump(final_pipe, bytes_io)
        bytes_io.seek(0)
        st.download_button("⬇️ Download trained model (.pkl)", data=bytes_io, file_name=f"leadscorer_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")

        st.caption("You can upload a saved .pkl below and score new leads without retraining.")

        # Upload model for scoring new leads
        st.subheader("7) Score new leads")
        up_model = st.file_uploader("Upload trained model (.pkl)", type=["pkl"], key="mdl_up")
        if up_model:
            try:
                loaded_pipe = joblib.load(up_model)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                loaded_pipe = None
        else:
            loaded_pipe = final_pipe

        new_src = st.radio("New leads source", ["Upload CSV","Google Sheets URL"])
        if new_src == "Upload CSV":
            new_file = st.file_uploader("New leads CSV (no label required)", type=["csv"], key="newcsv")
            df_new = pd.read_csv(new_file) if new_file else None
        else:
            new_url = st.text_input("Paste Google Sheets CSV URL for NEW leads")
            df_new = pd.read_csv(new_url) if new_url else None

        if df_new is not None and loaded_pipe is not None:
            st.write("**New leads preview:**")
            st.dataframe(df_new.head(10), use_container_width=True)
            # Align columns: keep only training feature columns (others ignored)
            feat_missing = [c for c in feature_cols if c not in df_new.columns]
            if feat_missing:
                st.info(f"Missing features in new leads: {feat_missing} — they will be imputed when possible.")
            X_new = df_new.reindex(columns=feature_cols, fill_value=np.nan)
            scores = loaded_pipe.predict_proba(X_new)[:, 1]
            out = df_new.copy()
            out["lead_score"] = scores
            out["recommended_action"] = np.where(out["lead_score"] >= float(best_row.threshold if "best_row" in locals() else thr_slider),
                                                 "PRIORITIZE", "LOW PRIORITY")
            st.success("Scoring complete.")
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button(
                "⬇️ Download scored leads (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name=f"scored_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

st.markdown("---")
st.caption("Notes: RandomizedSearchCV tunes key params; calibration improves probability quality; threshold chosen by minimizing business cost FP*cost_fp + FN*cost_fn.")
