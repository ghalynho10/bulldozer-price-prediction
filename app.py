import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bulldozer Price Predictor", page_icon="🚜", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_results():
    return pd.read_csv("data/val_results.csv")

results = load_results()

# ---------- Header ----------
st.title("🚜 Predicting Bulldozer Sale Prices with Machine Learning")
st.markdown(
    "An end-to-end regression project predicting used heavy-equipment auction prices "
    "from the Kaggle **Blue Book for Bulldozers** dataset (400,000+ auction records)."
)
st.markdown(
    "[GitHub Repo](https://github.com/ghalynho10) &nbsp;|&nbsp; "
    "Dataset: [Kaggle Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers)"
)
st.divider()

# ---------- Problem overview ----------
st.header("1. The Problem")
st.markdown(
    """
Given a bulldozer's characteristics (Size, configuration, sale date, etc.), predict its
sale price at auction. This is a **regression** task, evaluated using **RMSLE**
(Root Mean Squared Log Error) — the same metric used in the original Kaggle competition.

RMSLE penalizes relative error rather than absolute error, which suits a market where a
$1,000 miss on a $5,000 machine is far more significant than the same miss on a $100,000 machine.
"""
)

# ---------- Model comparison ----------
st.header("2. Model Performance")

col1, col2 = st.columns(2)

metrics_df = pd.DataFrame(
    {
        "Model": ["RandomizedSearchCV (10k subset)", "Final Model (full data, tuned params)"],
        "Training RMSLE": [0.271, 0.137],
        "Valid RMSLE": [0.298, 0.244],
        "Training MAE ($)": [5949, 2714],
        "Valid MAE ($)": [7346, 5922],
        "Training R²": [0.841, 0.963],
        "Valid R²": [0.826, 0.882],
    }
)

with col1:
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Training RMSLE", x=metrics_df["Model"], y=metrics_df["Training RMSLE"]))
    fig.add_trace(go.Bar(name="Valid RMSLE", x=metrics_df["Model"], y=metrics_df["Valid RMSLE"]))
    fig.update_layout(barmode="group", yaxis_title="RMSLE (lower is better)", height=350,
                       margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.info(
    "Training the final model on the **full dataset** (rather than the 10,000-row subset used "
    "for hyperparameter search) improved validation RMSLE by ~18% and cut validation MAE by "
    "over $1,400 — confirming a two-stage tuning strategy (cheap search on a subset, retrain on "
    "everything) as an efficient approach for large tabular datasets."
)

# ---------- Predicted vs Actual ----------
st.header("3. Predicted vs. Actual Sale Price")
st.markdown("Validation set predictions (n = {:,}) from the final model, plotted against actual sale prices.".format(len(results)))

size_order = ["Compact", "Mini", "Small", "Medium", "Large / Medium", "Large", "Unknown"]
available_sizes = [s for s in size_order if s in results["ProductSize"].unique()]
selected_sizes = st.multiselect("Filter by Product Size", available_sizes, default=available_sizes)
filtered = results[results["ProductSize"].isin(selected_sizes)]
st.caption(f"Showing {len(filtered):,} of {len(results):,} validation predictions.")

fig2 = px.scatter(
    filtered, x="actual", y="predicted", color="ProductSize",
    category_orders={"ProductSize": size_order},
    opacity=0.4,
    labels={"actual": "Actual Sale Price ($)", "predicted": "Predicted Sale Price ($)"},
    height=550,
)
max_val = max(results["actual"].max(), results["predicted"].max())
fig2.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                           line=dict(color="black", dash="dash"), name="Perfect prediction"))
fig2.update_layout(showlegend=True)
st.plotly_chart(fig2, use_container_width=True)

# Error by size table
st.subheader("Error by product size")
err_df = results.copy()
err_df["abs_error"] = (err_df["actual"] - err_df["predicted"]).abs()
err_df["pct_error"] = err_df["abs_error"] / err_df["actual"] * 100
summary = (
    err_df.groupby("ProductSize")
    .agg(count=("actual", "size"), mean_abs_error=("abs_error", "mean"), mean_pct_error=("pct_error", "mean"))
    .reindex([s for s in size_order if s in err_df["ProductSize"].unique()])
)
summary.columns = ["Count", "Mean Abs Error ($)", "Mean % Error"]
st.dataframe(
    summary.style.format({"Mean Abs Error ($)": "${:,.0f}", "Mean % Error": "{:.1f}%"}),
    use_container_width=True,
)
st.caption(
    "Nearly half of validation rows have missing `ProductSize` data (labeled 'Unknown'), reflecting "
    "a real gap in the source dataset rather than a modeling artifact."
)

st.caption(
    "Points close to the red dashed line indicate accurate predictions. The model performs "
    "consistently across price ranges, with somewhat wider spread at higher price points — "
    "expected given fewer high-value auctions in the training data."
)

# ---------- Feature importance ----------
st.header("4. Feature Importance")

feat_df = pd.DataFrame(
    {
        "feature": ["YearMade", "ProductSize", "saleYear", "fiSecondaryDesc", "Enclosure",
                    "fiModelDesc", "fiProductClassDesc", "fiBaseModel", "ModelID", "SalesID"],
        "importance": [0.2071, 0.1555, 0.0732, 0.0656, 0.0552, 0.0436, 0.0436, 0.0434, 0.0394, 0.0349],
    }
).sort_values("importance")

fig3 = px.bar(feat_df, x="importance", y="feature", orientation="h",
              labels={"importance": "Feature Importance", "feature": ""}, height=450)
st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    """
**`YearMade`** dominates as the single strongest predictor — consistent with domain intuition,
since equipment age is the primary driver of resale value. **`ProductSize`** and **`saleYear`**
(capturing broader market pricing trends) round out the top drivers.

**Error is consistent across equipment sizes.** Despite `ProductSize` being the second-most
important feature for predicting price *level*, mean percentage error stays in a narrow 18-21%
band across all size categories (see Section 3) — the model isn't systematically worse at
predicting price for any particular equipment size.

**Limitation worth naming:** `SalesID` and `ModelID` appear among the top 10 despite being
identifier fields with no inherent causal relationship to price — likely reflecting incidental
correlations (e.g., IDs assigned roughly in time order) rather than genuine signal. This is a
candidate for removal in future iterations.
"""
)

# ---------- Conclusions ----------
st.header("5. Conclusions & Limitations")
st.markdown(
    """
- **Validation RMSLE of 0.244** corresponds to predictions typically within ~24-28% of actual
  sale price — a reasonably strong result given the inherent variance in used heavy-equipment
  auctions (condition, regional demand, seller reputation aren't fully captured in this dataset).
- There's a meaningful gap between training RMSLE (0.137) and validation RMSLE (0.244),
  indicating some overfitting. Further tuning (`max_depth`, `min_samples_leaf`) or gradient-boosted
  alternatives (XGBoost) could close this gap.
- **Next steps:** remove ID-like features contributing spurious importance, compare against
  XGBoost/LightGBM, and expand this demo with a full-feature prediction pipeline.
"""
)

st.caption("Built with scikit-learn (RandomForestRegressor) and Streamlit. Full analysis and code on GitHub.")