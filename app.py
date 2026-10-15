import streamlit as st
import pandas as pd

from kokumi_infer import load_model, predict_smiles

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

    results = []
    progress = st.progress(0)
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        name = row["Name"]
        smiles = row[smiles_col]
        try:
            probability, label = predict_smiles(model, smiles)
        except Exception:
            probability, label = None, "Invalid"
        results.append([name, smiles, probability, label])
        progress.progress((idx + 1) / total)

    result_df = pd.DataFrame(
        results,
        columns=["Name", "SMILES", "Probability", "Prediction"],
    )

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
