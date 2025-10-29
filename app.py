import io
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="LeadScorer — MVP", page_icon="📈", layout="wide")
st.title("📈 LeadScorer — AI Lead Scoring (MVP)")
st.caption("Upload a training CSV with historical outcomes to train a model, then score new leads — 100% in the browser.")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
1. **Upload training CSV** (must include a binary outcome column, e.g., `closed_loan`).
2. Pick the outcome column and feature columns.
3. Click **Train model** → see AUC and feature importances.
4. **Upload new leads CSV** (no label) → get **scores** to download.
    """)
    st.markdown("---")
    st.markdown("**CSV format tips:**")
    st.code("lead_id, channel, device, time_on_site, page_views, income, fico, dti, closed_loan")

st.subheader("1) Upload training data")
train_file = st.file_uploader("Training CSV (must contain the outcome/label column)", type=["csv"])

DEMO = st.checkbox("Use demo training data", value=(train_file is None))
if DEMO and not train_file:
    df_train = pd.read_csv("data/leads_training_sample.csv")
else:
    if not train_file:
        st.info("Upload a training CSV or toggle **Use demo training data**.")
        st.stop()
    df_train = pd.read_csv(train_file)

st.write("**Training data preview:**")
st.dataframe(df_train.head(10), use_container_width=True)

# pick target/outcome
all_cols = list(df_train.columns)
target_col = st.selectbox("Select the outcome/label column (binary: 0/1)", options=all_cols, index=max(0, all_cols.index("closed_loan")) if "closed_loan" in all_cols else 0)

feature_cols_default = [c for c in all_cols if c != target_col and c not in ("lead_id",)]
feature_cols = st.multiselect("Select feature columns (predictors)", options=[c for c in all_cols if c != target_col], default=feature_cols_default)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# Build X, y
df_model = df_train.dropna(subset=[target_col]).copy()
y = df_model[target_col].astype(int)
X = df_model[feature_cols]

# identify dtypes
cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Preprocess: impute + one hot for categoricals, impute for numerics
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

# Model
model = GradientBoostingClassifier(random_state=42)

pipe = Pipeline(steps=[("prep", preprocess), ("clf", model)])

st.subheader("2) Train & evaluate")
col_btn, col_split = st.columns([1,3])
with col_btn:
    test_size = st.slider("Test size (holdout %)", 0.1, 0.4, 0.25, 0.05)
train_btn = st.button("🚀 Train model")

if train_btn:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)

        # Evaluate
        prob_test = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob_test)
        st.success(f"Model trained — AUC: **{auc:.3f}**")

        # ROC curve
        fpr, tpr, thr = roc_curve(y_test, prob_test)
        fig = px.line(x=fpr, y=tpr, labels={"x":"False Positive Rate", "y":"True Positive Rate"},
                      title="ROC Curve")
        fig.add_hline(y=0, line_dash="dash")
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance (approx): pull transformed feature names
        oh = pipe.named_steps["prep"].transformers_[1][1].named_steps["oh"]
        cat_feature_names = list(oh.get_feature_names_out(cat_cols)) if cat_cols else []
        feature_names = num_cols + cat_feature_names

        importances = pipe.named_steps["clf"].feature_importances_
        fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        st.subheader("Feature importance (approx.)")
        st.dataframe(fi.head(30), use_container_width=True)
        fi_fig = px.bar(fi.head(20), x="importance", y="feature", orientation="h", title="Top Features")
        st.plotly_chart(fi_fig, use_container_width=True)

        st.subheader("3) Score new leads (optional)")
        new_file = st.file_uploader("Upload NEW leads CSV to score (no label needed)", type=["csv"], key="newcsv")
        DEMO2 = st.checkbox("Use demo NEW leads", value=(new_file is None))
        if DEMO2 and not new_file:
            df_new = pd.read_csv("data/leads_new_sample.csv")
        else:
            if new_file:
                df_new = pd.read_csv(new_file)
            else:
                df_new = None

        if df_new is not None:
            st.write("**New leads preview:**")
            st.dataframe(df_new.head(10), use_container_width=True)
            # Keep only known feature columns; unseen columns are ignored by the pipeline
            missing_feats = [c for c in feature_cols if c not in df_new.columns]
            if missing_feats:
                st.info(f"Missing features in new leads: {missing_feats} — they will be imputed where possible.")

            X_new = df_new.reindex(columns=feature_cols, fill_value=np.nan)
            scores = pipe.predict_proba(X_new)[:, 1]
            out = df_new.copy()
            out["lead_score"] = scores

            st.success("Scoring complete. Download below.")
            st.dataframe(out.head(20), use_container_width=True)

            # download button
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download scored leads (CSV)",
                data=csv_bytes,
                file_name=f"scored_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

st.markdown("---")
st.caption("Tip: Outcome should be binary (0/1). Categorical columns are one-hot encoded. Numeric columns are median-imputed. Model: Gradient Boosting Classifier.")
