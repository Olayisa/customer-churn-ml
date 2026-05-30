import streamlit as st
import numpy as npy
import pandas as pds
import matplotlib.pyplot as mpl
import seaborn as sbn
import shap
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

MODELS_DIR = "models"

# ── Load pre-trained models saved from the notebook ──────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    if not os.path.exists(MODELS_DIR):
        st.error(
            "**Models folder not found.**\n\n"
            "Please run all cells in `Customer_Churn_ML_Model.ipynb` first, "
            "especially **Section 12 — Save Models & Artifacts for Streamlit**."
        )
        st.stop()

    xgb      = joblib.load(f'{MODELS_DIR}/xgb_tuned.pkl')
    lgbm     = joblib.load(f'{MODELS_DIR}/lgb_tuned.pkl')
    scaler   = joblib.load(f'{MODELS_DIR}/scaler.pkl')
    X_test   = pds.read_csv(f'{MODELS_DIR}/X_test.csv')
    y_test   = pds.read_csv(f'{MODELS_DIR}/y_test.csv').squeeze()
    FEATURE_NAMES = joblib.load(f'{MODELS_DIR}/feature_names.pkl')

    explainer_xgb  = shap.TreeExplainer(xgb)
    explainer_lgbm = shap.TreeExplainer(lgbm)

    return {
        'xgb': xgb, 'lgbm': lgbm, 'scaler': scaler,
        'explainer_xgb': explainer_xgb, 'explainer_lgbm': explainer_lgbm,
        'X_test': X_test, 'y_test': y_test,
        'FEATURE_NAMES': FEATURE_NAMES,
    }

FEATURE_NAMES = [
    'tenure_months', 'monthly_charges', 'total_charges',
    'num_products', 'support_calls', 'payment_delay_days',
    'contract_length', 'online_security', 'tech_support',
    'streaming_tv', 'age', 'satisfaction_score',
    'data_usage_gb', 'late_payments', 'promo_discount'
]

data = load_models()
FEATURE_NAMES = data['FEATURE_NAMES']

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📉 Churn Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 Predict a Customer", "📊 Model Performance", "🧠 SHAP Explainability"]
)
model_choice = st.sidebar.selectbox("Model", ["XGBoost", "LightGBM"])
model     = data['xgb']    if model_choice == "XGBoost" else data['lgbm']
explainer = data['explainer_xgb'] if model_choice == "XGBoost" else data['explainer_lgbm']

# ── Helper ────────────────────────────────────────────────────────────────────
def metrics_row(model, X_te, y_te):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    return {
        'Accuracy' : accuracy_score(y_te, y_pred),
        'ROC-AUC'  : roc_auc_score(y_te, y_proba),
        'F1'       : f1_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred),
        'Recall'   : recall_score(y_te, y_pred),
    }

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Customer Churn Prediction")
    st.markdown(
        "This portfolio app trains **XGBoost** and **LightGBM** models on a synthetic "
        "telco-style dataset, then explains predictions with **SHAP**."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset size", "5,000 customers")
    col2.metric("Churn rate", f"{data['df']['churn'].mean():.1%}")
    col3.metric("Features", str(len(FEATURE_NAMES)))

    st.markdown("---")
    st.subheader("Feature descriptions")
    desc = {
        'tenure_months'      : 'Months the customer has been with the company',
        'monthly_charges'    : 'Current monthly bill ($)',
        'total_charges'      : 'Cumulative spend ($)',
        'num_products'       : 'Number of subscribed products',
        'support_calls'      : 'Support calls in the last 6 months',
        'payment_delay_days' : 'Average days late on payment',
        'contract_length'    : 'Contract duration (encoded)',
        'online_security'    : 'Has online security add-on',
        'tech_support'       : 'Has tech support add-on',
        'streaming_tv'       : 'Has streaming TV',
        'age'                : 'Customer age',
        'satisfaction_score' : 'Last survey score (1–5)',
        'data_usage_gb'      : 'Monthly data usage (GB)',
        'late_payments'      : 'Number of late payments (lifetime)',
        'promo_discount'     : 'Active promotional discount',
    }
    st.dataframe(
        pds.DataFrame(desc.items(), columns=['Feature', 'Description']),
        use_container_width=True, hide_index=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT A CUSTOMER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict a Customer":
    st.title("🔍 Predict Churn for a Customer")
    st.markdown("Adjust the sliders and click **Predict** to see the churn probability and SHAP explanation.")

    col_l, col_r = st.columns(2)
    with col_l:
        tenure          = st.slider("Tenure (months)",        0, 72, 12)
        monthly_charges = st.slider("Monthly charges ($)",    10, 150, 65)
        total_charges   = st.slider("Total charges ($)",      0, 10000, 800)
        num_products    = st.slider("Number of products",     1, 8, 2)
        support_calls   = st.slider("Support calls",          0, 20, 3)
        payment_delay   = st.slider("Payment delay (days)",   0, 30, 2)
        contract_length = st.slider("Contract length",        0, 2, 0)
        age             = st.slider("Age",                    18, 80, 35)
    with col_r:
        online_security  = st.selectbox("Online security",  [0, 1], format_func=lambda x: "Yes" if x else "No")
        tech_support     = st.selectbox("Tech support",     [0, 1], format_func=lambda x: "Yes" if x else "No")
        streaming_tv     = st.selectbox("Streaming TV",     [0, 1], format_func=lambda x: "Yes" if x else "No")
        satisfaction     = st.slider("Satisfaction score",  1.0, 5.0, 3.5, 0.1)
        data_usage       = st.slider("Data usage (GB)",     0, 100, 20)
        late_payments    = st.slider("Late payments",       0, 20, 1)
        promo_discount   = st.slider("Promo discount",      0.0, 1.0, 0.0, 0.05)

    input_df = pds.DataFrame([[
        tenure, monthly_charges, total_charges, num_products,
        support_calls, payment_delay, contract_length,
        online_security, tech_support, streaming_tv,
        age, satisfaction, data_usage, late_payments, promo_discount
    ]], columns=FEATURE_NAMES)

    if st.button("Predict", type="primary"):
        prob    = model.predict_proba(input_df)[0][1]
        pred    = int(prob >= 0.5)
        label   = "🔴 Likely to Churn" if pred else "🟢 Likely to Stay"
        delta   = f"{prob:.1%} churn probability"

        st.markdown("---")
        st.subheader("Prediction Result")
        st.metric(label="Outcome", value=label, delta=delta,
                  delta_color="inverse")

        st.markdown("#### SHAP Waterfall — why this prediction?")
        sv = explainer(input_df)
        fig, ax = mpl.subplots(figsize=(10, 5))
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        mpl.tight_layout()
        st.pyplot(fig)
        mpl.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    m_xgb  = metrics_row(data['xgb'],  data['X_test'], data['y_test'])
    m_lgbm = metrics_row(data['lgbm'], data['X_test'], data['y_test'])

    tab1, tab2 = st.tabs(["Metrics table", "Charts"])

    with tab1:
        cmp = pds.DataFrame([
            {'Model': 'XGBoost',  **m_xgb},
            {'Model': 'LightGBM', **m_lgbm},
        ]).set_index('Model').round(4)
        st.dataframe(cmp, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        # ── Grouped bar ──
        with col1:
            metrics_keys = ['Accuracy', 'ROC-AUC', 'F1', 'Precision', 'Recall']
            x   = npy.arange(len(metrics_keys))
            bw  = 0.35
            fig, ax = mpl.subplots(figsize=(7, 4))
            ax.bar(x - bw/2, [m_xgb[k]  for k in metrics_keys], bw, label='XGBoost',  color='#C44E52', alpha=0.85)
            ax.bar(x + bw/2, [m_lgbm[k] for k in metrics_keys], bw, label='LightGBM', color='#4C72B0', alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(metrics_keys, fontsize=9)
            ax.set_ylim(0, 1.1); ax.legend(); ax.set_title('Metric Comparison')
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig); mpl.close(fig)

        # ── ROC curves ──
        with col2:
            fig, ax = mpl.subplots(figsize=(6, 4))
            for name, mdl, color in [('XGBoost', data['xgb'], '#C44E52'),
                                      ('LightGBM', data['lgbm'], '#4C72B0')]:
                proba = mdl.predict_proba(data['X_test'])[:, 1]
                fpr, tpr, _ = roc_curve(data['y_test'], proba)
                auc = roc_auc_score(data['y_test'], proba)
                ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
            ax.plot([0,1],[0,1],'k--',lw=1)
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title('ROC Curves'); ax.legend(fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig); mpl.close(fig)

        # ── Confusion matrix ──
        st.markdown("#### Confusion Matrix")
        cm_col1, cm_col2 = st.columns(2)
        for col, (name, mdl) in zip([cm_col1, cm_col2],
                                     [('XGBoost', data['xgb']),
                                      ('LightGBM', data['lgbm'])]):
            with col:
                cm = confusion_matrix(data['y_test'], mdl.predict(data['X_test']))
                fig, ax = mpl.subplots(figsize=(4, 3))
                sbn.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Stay', 'Churn'],
                            yticklabels=['Stay', 'Churn'], cbar=False)
                ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
                mpl.tight_layout()
                st.pyplot(fig); mpl.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Explainability")
    st.markdown(f"Using **{model_choice}** — switch models in the sidebar.")

    @st.cache_resource(show_spinner="Computing SHAP values...")
    def get_shap(model_name):
        mdl = data['xgb'] if model_name == "XGBoost" else data['lgbm']
        exp = data['explainer_xgb'] if model_name == "XGBoost" else data['explainer_lgbm']
        return exp(data['X_test'])

    shap_vals = get_shap(model_choice)

    tab_bee, tab_bar, tab_water = st.tabs(["Beeswarm", "Bar (mean |SHAP|)", "Waterfall"])

    with tab_bee:
        st.markdown("Each dot is one customer. Color = feature value (red=high, blue=low). X-axis = impact on prediction.")
        fig, _ = mpl.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_vals, max_display=15, show=False)
        mpl.tight_layout()
        st.pyplot(fig); mpl.close(fig)

    with tab_bar:
        st.markdown("Average absolute SHAP value per feature — the higher, the more influential globally.")
        fig, _ = mpl.subplots(figsize=(9, 6))
        shap.plots.bar(shap_vals, max_display=15, show=False)
        mpl.tight_layout()
        st.pyplot(fig); mpl.close(fig)

    with tab_water:
        st.markdown("Step-by-step explanation for a single customer prediction.")
        idx = st.number_input(
            "Customer index (0 – 999)", min_value=0,
            max_value=len(data['X_test']) - 1, value=0, step=1
        )
        actual = data['y_test'].iloc[int(idx)]
        pred   = model.predict(data['X_test'].iloc[[int(idx)]])[0]
        prob   = model.predict_proba(data['X_test'].iloc[[int(idx)]])[0][1]
        st.markdown(
            f"**Actual:** {'Churn' if actual else 'Stay'} &nbsp;|&nbsp; "
            f"**Predicted:** {'Churn' if pred else 'Stay'} &nbsp;|&nbsp; "
            f"**Churn probability:** {prob:.1%}"
        )
        fig, _ = mpl.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_vals[int(idx)], max_display=12, show=False)
        mpl.tight_layout()
        st.pyplot(fig); mpl.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("Built with XGBoost · LightGBM · SHAP · Streamlit")
