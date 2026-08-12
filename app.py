import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Enterprise Retail Intelligence Platform",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Enterprise Retail Intelligence Platform")
st.subheader("Sales Prediction & Inventory Intelligence Dashboard")


# ==========================================================
# FILE PATHS
# ==========================================================

pipeline_path = "models/xgboost_prediction_pipeline.pkl"
prediction_features_path = "data/dashboard_prediction_features.csv"
prediction_inputs_path = "data/dashboard_prediction_inputs.csv"
inventory_path = "data/processed/inventory_optimization.csv"
feature_importance_path = "data/processed/xgboost_feature_importance.csv"


# ==========================================================
# CHECK REQUIRED FILES
# ==========================================================

required_files = {
    "XGBoost Pipeline": pipeline_path,
    "Prediction Features": prediction_features_path,
    "Prediction Inputs": prediction_inputs_path,
    "Inventory Optimization": inventory_path,
    "Feature Importance": feature_importance_path
}

missing_files = []

for file_name, file_path in required_files.items():
    if not os.path.exists(file_path):
        missing_files.append(f"{file_name}: {file_path}")


if missing_files:

    st.error("Some required project files are missing:")

    for file in missing_files:
        st.write(f"❌ {file}")

    st.stop()


# ==========================================================
# LOAD XGBOOST PREDICTION PIPELINE
# ==========================================================

pipeline = joblib.load(pipeline_path)

model = pipeline["model"]
features = pipeline["features"]


# ==========================================================
# LOAD DASHBOARD DATA
# ==========================================================

dashboard_data = pd.read_csv(
    prediction_features_path
)

display_data = pd.read_csv(
    prediction_inputs_path
)


# ==========================================================
# LOAD INVENTORY OPTIMIZATION DATA
# ==========================================================

inventory_data = pd.read_csv(
    inventory_path
)


# ==========================================================
# LOAD FEATURE IMPORTANCE DATA
# ==========================================================

feature_importance_data = pd.read_csv(
    feature_importance_path
)


st.success(
    "Prediction Model, Inventory Data and Explainable AI Data Loaded Successfully!"
)


# ==========================================================
# KPI SUMMARY
# ==========================================================

st.subheader("📊 Prediction KPI Summary")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


# ----------------------------------------------------------
# Generate predictions for dashboard records
# ----------------------------------------------------------

try:

    all_predictions = model.predict(
        dashboard_data[features]
    )

    all_predictions = pd.Series(
        all_predictions
    )

except Exception as e:

    st.warning(
        f"Unable to generate overall predictions: {e}"
    )

    all_predictions = pd.Series(dtype=float)


with kpi1:

    st.metric(
        "Total Predictions",
        len(all_predictions)
    )


with kpi2:

    if len(all_predictions) > 0:

        st.metric(
            "Average Predicted Sales",
            f"{all_predictions.mean():.2f}"
        )

    else:

        st.metric(
            "Average Predicted Sales",
            "N/A"
        )


with kpi3:

    if len(all_predictions) > 0:

        st.metric(
            "Highest Predicted Sales",
            f"{all_predictions.max():.2f}"
        )

    else:

        st.metric(
            "Highest Predicted Sales",
            "N/A"
        )


with kpi4:

    if len(all_predictions) > 0:

        st.metric(
            "Lowest Predicted Sales",
            f"{all_predictions.min():.2f}"
        )

    else:

        st.metric(
            "Lowest Predicted Sales",
            "N/A"
        )


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.subheader("🤖 Model Information")

model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    st.metric(
        "Model",
        "XGBoost"
    )


with model_col2:

    st.metric(
        "Features",
        len(features)
    )


with model_col3:

    st.metric(
        "Prediction Records",
        len(dashboard_data)
    )


# ==========================================================
# AVAILABLE PRODUCT / STORE DATA
# ==========================================================

st.subheader("📦 Available Product / Store Data")

st.dataframe(
    dashboard_data.head(10),
    use_container_width=True
)


# ==========================================================
# SALES PREDICTION
# ==========================================================

st.subheader("🔮 Sales Prediction")


selected_item = st.selectbox(
    "Select Item",
    display_data["item_id"].unique()
)


selected_store = st.selectbox(
    "Select Store",
    display_data["store_id"].unique()
)


# ==========================================================
# FIND MATCHING PRODUCT / STORE
# ==========================================================

selected_data = display_data[
    (display_data["item_id"] == selected_item)
    &
    (display_data["store_id"] == selected_store)
]


if len(selected_data) > 0:

    st.write("### Selected Product / Store")

    selected_index = selected_data.index[0]


    # ------------------------------------------------------
    # Get corresponding model input
    # ------------------------------------------------------

    prediction_input = dashboard_data.loc[
        [selected_index],
        features
    ]


    st.dataframe(
        selected_data,
        use_container_width=True
    )


    # ------------------------------------------------------
    # Generate Prediction
    # ------------------------------------------------------

    if st.button(
        "🔮 Predict Sales",
        key="predict_sales_button"
    ):

        prediction = model.predict(
            prediction_input
        )

        predicted_sales = float(
            prediction[0]
        )


        st.success(
            "Prediction Generated Successfully!"
        )


        prediction_col1, prediction_col2 = st.columns(2)


        with prediction_col1:

            st.metric(
                "Predicted Sales",
                f"{predicted_sales:.2f}"
            )


        with prediction_col2:

            st.metric(
                "Selected Item",
                selected_item
            )


        # --------------------------------------------------
        # Sales Interpretation
        # --------------------------------------------------

        if predicted_sales < 1:

            st.info(
                "📉 Low expected sales for this item-store combination."
            )

            st.write(
                "💡 Recommendation: Consider maintaining lower "
                "inventory levels and reviewing demand for this product."
            )


        elif predicted_sales < 5:

            st.warning(
                "📊 Moderate expected sales for this item-store combination."
            )

            st.write(
                "💡 Recommendation: Maintain normal inventory levels "
                "and monitor sales performance."
            )


        else:

            st.success(
                "📈 High expected sales for this item-store combination."
            )

            st.write(
                "💡 Recommendation: Consider increasing inventory "
                "to meet expected demand."
            )


else:

    st.warning(
        "No matching product and store combination found."
    )


# ==========================================================
# INVENTORY OPTIMIZATION
# ==========================================================

st.divider()

st.subheader("📦 Inventory Optimization")


inventory_col1, inventory_col2, inventory_col3 = st.columns(3)


with inventory_col1:

    st.metric(
        "Inventory Records",
        len(inventory_data)
    )


with inventory_col2:

    st.metric(
        "Average Safety Stock",
        f"{inventory_data['safety_stock'].mean():.2f}"
    )


with inventory_col3:

    st.metric(
        "Average Reorder Point",
        f"{inventory_data['reorder_point'].mean():.2f}"
    )


# ==========================================================
# INVENTORY PRODUCT / STORE SELECTION
# ==========================================================

inventory_item = st.selectbox(
    "Select Inventory Item",
    inventory_data["item_id"].unique(),
    key="inventory_item"
)


inventory_store = st.selectbox(
    "Select Inventory Store",
    inventory_data["store_id"].unique(),
    key="inventory_store"
)


inventory_selection = inventory_data[
    (inventory_data["item_id"] == inventory_item)
    &
    (inventory_data["store_id"] == inventory_store)
]


if len(inventory_selection) > 0:

    inventory_row = inventory_selection.iloc[0]


    inv1, inv2, inv3, inv4 = st.columns(4)


    with inv1:

        st.metric(
            "Average Daily Sales",
            f"{inventory_row['average_daily_sales']:.2f}"
        )


    with inv2:

        st.metric(
            "Maximum Daily Sales",
            f"{inventory_row['maximum_daily_sales']:.2f}"
        )


    with inv3:

        st.metric(
            "Safety Stock",
            f"{inventory_row['safety_stock']:.2f}"
        )


    with inv4:

        st.metric(
            "Reorder Point",
            f"{inventory_row['reorder_point']:.2f}"
        )


    st.dataframe(
        inventory_selection,
        use_container_width=True
    )


    st.info(
        f"💡 Recommended action: When inventory for "
        f"{inventory_item} at {inventory_store} reaches approximately "
        f"{inventory_row['reorder_point']:.0f} units, consider placing "
        f"a replenishment order."
    )


# ==========================================================
# TOP PRODUCTS BY REORDER POINT
# ==========================================================

st.subheader("📈 Top Products by Reorder Point")


top_reorder = inventory_data.sort_values(
    by="reorder_point",
    ascending=False
).head(10)


st.dataframe(
    top_reorder[
        [
            "item_id",
            "store_id",
            "average_daily_sales",
            "safety_stock",
            "reorder_point"
        ]
    ],
    use_container_width=True
)


# ==========================================================
# REORDER POINT CHART
# ==========================================================

st.subheader("📊 Top 10 Reorder Points")


fig, ax = plt.subplots(
    figsize=(12, 6)
)


ax.bar(
    top_reorder["item_id"].astype(str),
    top_reorder["reorder_point"]
)


ax.set_xlabel(
    "Product"
)

ax.set_ylabel(
    "Reorder Point"
)

ax.set_title(
    "Top 10 Products by Reorder Point"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ==========================================================
# EXPLAINABLE AI
# ==========================================================

st.divider()

st.subheader("🧠 Explainable AI — XGBoost Feature Importance")


# ----------------------------------------------------------
# Top 10 features
# ----------------------------------------------------------

top_features = feature_importance_data.head(10)


# ----------------------------------------------------------
# Feature Importance Metrics
# ----------------------------------------------------------

ai_col1, ai_col2 = st.columns(2)


with ai_col1:

    st.metric(
        "Total Model Features",
        len(feature_importance_data)
    )


with ai_col2:

    st.metric(
        "Most Important Feature",
        top_features.iloc[0]["Feature"]
    )


# ----------------------------------------------------------
# Feature Importance Table
# ----------------------------------------------------------

st.write("### Top 10 Important Features")

st.dataframe(
    top_features,
    use_container_width=True
)


# ==========================================================
# FEATURE IMPORTANCE CHART
# ==========================================================

st.write("### Top 10 Feature Importance")


fig2, ax2 = plt.subplots(
    figsize=(10, 6)
)


ax2.barh(
    top_features["Feature"],
    top_features["Importance"]
)


ax2.set_xlabel(
    "Importance"
)

ax2.set_ylabel(
    "Feature"
)

ax2.set_title(
    "Top 10 XGBoost Feature Importance"
)


ax2.invert_yaxis()

plt.tight_layout()

st.pyplot(fig2)

plt.close(fig2)


# ==========================================================
# EXPLAINABLE AI INTERPRETATION
# ==========================================================

top_feature = top_features.iloc[0]


st.info(
    f"🔍 The most important feature is "
    f"{top_feature['Feature']} with an importance score of "
    f"{top_feature['Importance']:.4f}. "
    f"The feature importance values show which input features "
    f"contribute most to the XGBoost model's predictions."
)


