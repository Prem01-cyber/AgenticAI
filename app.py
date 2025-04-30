import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from main import FUNCTION_REGISTRY
import chardet

st.set_page_config(page_title="Agentic ML Copilot", layout="wide")
st.title("Agentic ML Assistant")

with open("metadata.json", "r") as f:
    metadata = json.load(f)

if "csv_file_path" not in st.session_state:
    st.session_state.csv_file_path = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "interaction_history" not in st.session_state:
    st.session_state.interaction_history = []

st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    # Read file bytes to detect encoding
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result["encoding"] or "utf-8"

    try:
        # Reset pointer and load CSV with detected encoding
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding)
        st.session_state.uploaded_df = df
        temp_path = "temp_input.csv"
        df.to_csv(temp_path, index=False)
        st.session_state.csv_file_path = temp_path
        st.success("✅ CSV uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")

if st.session_state.uploaded_df is not None:
    st.subheader("Uploaded CSV Preview")
    st.dataframe(st.session_state.uploaded_df.head())

st.sidebar.header("Select Task")
task = st.sidebar.selectbox("Available Functions", list(metadata.keys()))
st.header(f"Task: `{task}`")

payload_template = metadata[task]["sample_payload"]
user_payload = {}

st.subheader("Input Parameters")
for key, val in payload_template.items():
    if key == "csv_file":
        user_payload[key] = st.session_state.csv_file_path

    elif key in ["x_col", "y_col", "target"] and st.session_state.uploaded_df is not None:
        user_payload[key] = st.selectbox(f"{key}", st.session_state.uploaded_df.columns)

    elif key == "features" and st.session_state.uploaded_df is not None:
        user_payload[key] = st.multiselect(f"{key}", st.session_state.uploaded_df.columns)

    elif isinstance(val, str):
        user_payload[key] = st.text_input(key, val)
    elif isinstance(val, float):
        user_payload[key] = st.number_input(key, value=val)
    elif isinstance(val, int):
        user_payload[key] = st.number_input(key, value=val, step=1)
    elif isinstance(val, list):
        try:
            user_payload[key] = json.loads(st.text_area(f"{key} (JSON)", json.dumps(val, indent=2)))
        except:
            st.error(f"Invalid JSON for {key}")
            user_payload[key] = val

if st.button("Run Task"):
    try:
        result = FUNCTION_REGISTRY[task](user_payload, {}, [])
        st.session_state.interaction_history.append({
            "task": task,
            "input": user_payload,
            "result": result,
            "timestamp": str(datetime.now())
        })

        st.subheader("Result")
        if "output_file" in result:
            if result["output_file"].endswith(".csv"):
                df_result = pd.read_csv(result["output_file"])
                st.dataframe(df_result.head())
                with open(result["output_file"], "rb") as f:
                    st.download_button("Download CSV", f, file_name=result["output_file"])
            elif result["output_file"].endswith(".png"):
                st.image(result["output_file"])
        elif "recommended_model" in result:
            # Case where result is already parsed
            st.success(f"Model: {result['recommended_model']}")
            st.markdown(f"**Reason**: {result['reason']}")

        elif "response" in result:
            try:
                # Attempt to parse the stringified JSON inside 'response'
                parsed_response = json.loads(result["response"])
                if "recommended_model" in parsed_response:
                    st.success(f"Model: {parsed_response['recommended_model']}")
                    st.markdown(f"**Reason**: {parsed_response['reason']}")
                else:
                    st.json(parsed_response)
            except json.JSONDecodeError:
                st.warning("LLM returned non-JSON response:")
                st.text(result["response"])


    except Exception as e:
        st.error(f"❌ Error while running: {e}")

# Sidebar History
st.sidebar.markdown("---")
st.sidebar.subheader("🕓 Recent Runs")
for interaction in reversed(st.session_state.interaction_history[-5:]):
    st.sidebar.markdown(
        f"""
        <div style='margin-bottom: 10px;'>
            <strong>✅ {interaction['task']}</strong><br/>
            <span style='font-size: 0.8em; color: gray;'>{interaction['timestamp']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
