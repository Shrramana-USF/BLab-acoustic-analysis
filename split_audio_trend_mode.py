import streamlit as st
import pandas as pd
import io
from analysis_utils import (
    get_box_client,
    BASE_FOLDER_ID,
    ensure_task_folder,
    read_byte_stream,
)

TASKS = [
    "Rainbow passage",
    "Maximum sustained phonation on 'aaah'",
    "Comfortable sustained phonation on 'eeee'",
    "Glide up to your highest pitch on 'eeee'",
    "Glide down to your lowest pitch on 'eeee'",
    "Sustained 'aaah' at minimum volume",
    "Maximum loudness level (brief 'AAAH')",
    "Conversational speech",
]


def split_audio_trend_tab(_):
    st.subheader("Split Audio Trend")

    recorder_email = st.text_input("Recorder Email ID", key="trend_recorder_email")
    pid = st.text_input("PID (Unique Patient/Session ID)", key="trend_pid")
    task = st.selectbox("Select Task", TASKS, key="trend_task")

    # --- Basic validation ---
    if not (recorder_email and pid and task):
        st.info("Please fill in all fields to continue.")
        return

    client = get_box_client()

    # --- Locate Box folder hierarchy ---
    try:
        recorder_folder_id = ensure_task_folder(client, BASE_FOLDER_ID, recorder_email)
        pid_folder_id = ensure_task_folder(client, recorder_folder_id, pid)
        task_folder_id = ensure_task_folder(client, pid_folder_id, task)
    except Exception as e:
        st.error(f"Could not locate task folder: {e}")
        return

    st.info("Click 'Generate Trend' to analyse feature trends for this task.")
    load_trend = st.button("Generate Trend", key=f"generate_trend_{task}")
    if not load_trend:
        return

    # --- Gather feature CSVs ---
    try:
        items = client.folders.get_folder_items(str(task_folder_id))
        feature_files = [
            f for f in items.entries
            if f.type == "file" and f.name.endswith("_features.csv")
        ]
    except Exception as e:
        st.error(f"Error fetching folder items: {e}")
        return

    if not feature_files:
        st.warning("No feature CSV files found for this task.")
        return

    all_data = []
    for file_item in feature_files:
        try:
            # Expected filename: YYYY-MM-DD_<task>_features.csv
            date_str = file_item.name.split("_")[0]
            byte_stream = client.downloads.download_file(file_item.id)
            file_bytes = read_byte_stream(byte_stream)
            df = pd.read_csv(io.BytesIO(file_bytes))
            df["session"] = date_str
            all_data.append(df)
        except Exception as e:
            st.warning(f"Could not read {file_item.name}: {e}")

    if not all_data:
        st.warning("Could not load any valid feature files.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["Value"] = pd.to_numeric(combined_df["Value"], errors="coerce")

    # --- Display feature table ---
    pivot = combined_df.pivot_table(
        index="Feature", columns="session", values="Value", aggfunc="first"
    )
    st.dataframe(pivot, use_container_width=True)

    # --- Trend charts for each feature ---
    features = combined_df["Feature"].unique()
    for feat in features:
        feat_df = combined_df[combined_df["Feature"] == feat].sort_values("session")
        if not feat_df.empty:
            st.caption(f"Trend for {feat}")
            st.line_chart(feat_df.set_index("session")["Value"], use_container_width=True)
