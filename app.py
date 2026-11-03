import streamlit as st
import pandas as pd

import numpy as np

from kokumi_infer import load_model, predict_frame

st.set_page_config(
    page_title="Kokumi",
    page_icon="🧬",
    layout="centered",
)

st.title("Kokumi Batch Prediction")


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if "canonical SMILES" in df.columns:
        smiles_col = "canonical SMILES"
    elif "SMILES" in df.columns:
        smiles_col = "SMILES"
    else:
        st.error("CSV must contain 'SMILES' or 'canonical SMILES'")
        st.stop()

    if "Name" not in df.columns:
        df["Name"] = "Unknown"

    if len(df) == 0:
        st.warning("The CSV file has no rows.")
        st.stop()

    work = pd.DataFrame({
        "name": df["Name"].astype(str),
        "smiles": df[smiles_col].map(lambda value: "" if pd.isna(value) else str(value).strip()),
    })
    with st.spinner("Predicting..."):
        try:
            predicted = predict_frame(model, work)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    prediction = np.where(
        predicted["pred_class_clf"].eq("high"),
        "Kokumi",
        np.where(predicted["pred_class_clf"].eq("low"), "Non-Kokumi", "Invalid"),
    )
    result_df = pd.DataFrame({
        "Name": predicted["name"],
        "SMILES": predicted["smiles"],
        "Probability": predicted["prob_high"],
        "Prediction": prediction,
    })

    st.success("Prediction Completed")
    st.dataframe(result_df, width="stretch")

    st.subheader("Top 10 Kokumi Candidates")
    valid_df = result_df.dropna(subset=["Probability"])
    st.dataframe(valid_df.sort_values("Probability", ascending=False).head(10))

    st.subheader("Prediction Summary")
    st.write(result_df["Prediction"].value_counts())

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name="kokumi_prediction.csv",
        mime="text/csv",
    )
