import os
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "customer_data.csv")

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ E-Commerce Customer Segmentation")
st.caption("Simple ML project using customer purchasing behavior")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    df = pd.read_csv(os.path.join(MODEL_DIR, "clustered_customers.csv"))
    profile = pd.read_csv(os.path.join(MODEL_DIR, "segment_profile.csv"))
except Exception:
    st.error("Model files are missing. Run `python train_model.py` from the project root first.")
    st.stop()

# Friendly segment descriptions based on actual segment profiles.
means = df.groupby("Cluster")[features].mean()
rank_spend = means["TotalSpending"].rank(ascending=False)
rank_recency = means["Recency"].rank(ascending=True)

def segment_name(cluster):
    spend_rank = rank_spend[cluster]
    rec_rank = rank_recency[cluster]
    if spend_rank == 1 and rec_rank == 1:
        return "High-Value / Recent"
    if spend_rank == len(means) and rec_rank == len(means):
        return "Low-Value / Inactive"
    return "Regular / Mid-Value"

df["Segment"] = df["Cluster"].map(segment_name)

c1, c2, c3 = st.columns(3)
c1.metric("Customers", len(df))
c2.metric("Segments", df["Cluster"].nunique())
c3.metric("Avg. Spending", f"₹{df['TotalSpending'].mean():,.0f}")

st.subheader("Segment Distribution")
counts = df["Segment"].value_counts()
st.bar_chart(counts)

st.subheader("Segment Profiles")
display_profile = means.copy()
display_profile.index = [segment_name(i) for i in display_profile.index]
st.dataframe(display_profile.round(2), use_container_width=True)

st.divider()
st.subheader("Predict a Customer Segment")

col1, col2 = st.columns(2)
with col1:
    spending = st.number_input("Total Spending (₹)", min_value=0.0, value=5000.0, step=500.0)
    purchases = st.number_input("Number of Purchases", min_value=1, value=10, step=1)
    frequency = st.number_input("Purchase Frequency", min_value=0.0, value=0.5, step=0.05)
with col2:
    aov = st.number_input("Average Order Value (₹)", min_value=0.0, value=700.0, step=50.0)
    recency = st.number_input("Recency (days)", min_value=0, value=30, step=5)

if st.button("Predict Customer Segment", type="primary"):
    row = pd.DataFrame([[spending, purchases, frequency, aov, recency]], columns=features)
    scaled = scaler.transform(row)
    cluster = int(model.predict(scaled)[0])
    name = segment_name(cluster)
    st.success(f"Predicted Segment: **{name}**")
    st.write("This segment is determined from similarity to the learned customer clusters.")
