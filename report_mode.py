import streamlit as st
import pandas as pd
from analysis_utils import *
import matplotlib.pyplot as plt

def report_tab(folder_id):
    # --- Task selection ---
    st.subheader("Report and Analysis by Task")

    tasks = [ "Rainbow passage", "Maximum sustained phonation on 'aaah'", "Comfortable sustained phonation on 'eeee'", 
             "Glide up to your highest pitch on 'eeee'", "Glide down to your lowest pitch on 'eeee'", 
             "Sustained 'aaah' at minimum volume", "Maximum loudness level (brief 'AAAH')", "Conversational speech"]
    
    selected_task = st.radio(
        "Select a task to view reports:",
        options=tasks,
        index=None,
        horizontal=True,
        key="report_task_radio"
    )

    # --- Reset state when task changes ---
    if "prev_task_report" not in st.session_state:
        st.session_state.prev_task_report = None

    if selected_task != st.session_state.prev_task_report:
        st.session_state.prev_task_report = selected_task
        st.rerun()

    # --- Stop until a task is selected ---
    if selected_task is None:
        st.info("Please select a task to generate the report.")
        return

    # --- Check if task folder exists ---
    client = get_box_client()
    user_folder_id = str(folder_id)

    # Try to locate the matching task folder
    try:
        user_items = client.folders.get_folder_items(user_folder_id)
        task_folder = next(
            (f for f in user_items.entries if f.type == "folder" and f.name == selected_task),
            None
        )
    except Exception:
        st.error("Unable to access user folder.")
        return

    if not task_folder:
        st.warning("No sessions found for this task.")
        return

    task_folder_id = task_folder.id

    # --- Generate report only after task is selected ---
    st.info("Click 'Generate report' to analyse session trends for this task.")
    load_report = st.button("Generate Report", key=f"generate_report_{selected_task}")
    if not load_report:
        return

    # ---------------- REPORT MODE ----------------
    df, audio_map = fetch_all_features(client, str(task_folder_id))

    if df.empty:
        st.warning("No features found. Run some analyses first.")
        return

    st.subheader(f"Feature analysis across sessions for {selected_task}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    pivot = df.pivot_table(index="Feature", columns="session", values="Value", aggfunc="first")
    st.dataframe(pivot, use_container_width=True)

    features = df["Feature"].unique()
    for feat in features:
        feat_df = df[df["Feature"] == feat].sort_values("session")
        if not feat_df.empty:
            st.caption(f"Trend for {feat}")
            st.line_chart(feat_df.set_index("session")["Value"], use_container_width=True)

    # --- Sidebar audio playback ---
    st.sidebar.subheader(f"Session Audio — {selected_task}")
    for session_name in df["session"].unique():
        st.sidebar.markdown(f"**{session_name}**")
        audio_file_id = audio_map.get(session_name)
        if audio_file_id:
            byte_stream = client.downloads.download_file(audio_file_id)
            wav_bytes = read_byte_stream(byte_stream)
            st.sidebar.audio(wav_bytes, format="audio/wav")
        else:
            st.sidebar.warning("No audio found for this session")
