"""Streamlit demo.  Run:  streamlit run app.py   (train first: python train.py)"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# the pipeline pickles FeatureEngineer from train.py, so it must be importable
from train import FeatureEngineer  # noqa: F401

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="India House Price Estimator", page_icon="🏠", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return model, meta, metrics


model, meta, metrics = load_assets()
test_m = metrics["test"][metrics["selected_model"]]

st.title("🏠 India House Price Estimator")
st.caption("Multi-city ML valuation (Delhi, Mumbai, Bangalore, Hyderabad, Chennai, Kolkata). "
           "Trained on a synthetic dataset - for demonstration only.")

tab_pred, tab_perf = st.tabs(["Estimate price", "Model performance"])

with tab_pred:
    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("City", meta["cities"])
        locality = st.selectbox("Locality", meta["localities"][city])
        furnishing = st.selectbox("Furnishing", meta["furnishing"], index=1)
    with c2:
        bhk = st.slider("BHK", 1, 5, 2)
        area = st.number_input("Area (sq ft)", 300, 6000, 1200, step=50)
        age = st.slider("Property age (years)", 0, 30, 5)
    with c3:
        total_floors = st.number_input("Total floors in building", 1, 60, 12)
        floor = st.number_input("Floor", 0, 60, 4)

    stats = meta["locality_stats"][f"{city}|{locality}"]
    with st.expander("Location details (auto-filled from locality, editable)"):
        d1, d2 = st.columns(2)
        cbd = d1.number_input("Distance to CBD (km)", 0.0, 60.0, round(stats["distance_to_cbd_km"], 2))
        metro = d2.number_input("Distance to metro (km)", 0.0, 15.0, round(stats["distance_to_metro_km"], 2))

    if floor > total_floors:
        st.warning("Floor cannot be higher than total floors.")
    elif st.button("Estimate price", type="primary"):
        row = pd.DataFrame([{
            "city": city, "locality": locality, "BHK": bhk, "area": area, "floor": floor,
            "total_floors": total_floors, "age": age, "furnishing": furnishing,
            "latitude": stats["latitude"], "longitude": stats["longitude"],
            "distance_to_cbd_km": cbd, "distance_to_metro_km": metro,
        }])
        price = float(model.predict(row)[0])
        # rough range using the model's test MAPE
        pct = test_m["MAPE_%"] / 100
        lo, hi = price * (1 - pct), price * (1 + pct)

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated price", f"₹ {price:,.1f} Lakhs", f"≈ ₹ {price/100:,.2f} Cr")
        m2.metric("Likely range", f"{lo:,.0f} - {hi:,.0f} Lakhs")
        m3.metric("Price per sq ft", f"₹ {price*1e5/area:,.0f}")

with tab_perf:
    st.subheader(f"Selected model: {metrics['selected_model']}")
    a, b, c, d = st.columns(4)
    a.metric("R²", f"{test_m['R2']:.3f}")
    b.metric("MAE", f"{test_m['MAE_lakhs']:.1f} L")
    c.metric("RMSE", f"{test_m['RMSE_lakhs']:.1f} L")
    d.metric("MAPE", f"{test_m['MAPE_%']:.1f}%")
    st.markdown("**All models (held-out test set)**")
    st.dataframe(pd.DataFrame(metrics["test"]).T.round(3), use_container_width=True)
    p1, p2 = st.columns(2)
    p1.image(str(MODEL_DIR / "actual_vs_pred.png"), caption="Actual vs predicted")
    p2.image(str(MODEL_DIR / "feature_importance.png"), caption="Feature importance")
