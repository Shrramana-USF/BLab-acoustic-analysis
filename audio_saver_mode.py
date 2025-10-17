import streamlit as st
import os, io
import numpy as np
import soundfile as sf
from streamlit_advanced_audio import audix
from analysis_utils import (
    get_box_client,
    ensure_task_folder,
    read_audio_bytes,
    save_temp_mono_wav,
    upload_to_user_box,
)

# Define all tasks
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

def audio_saver_tab(folder_id):
    """
    Audio Saver tab (per-user).
    Receives date, PID, and an audio file.
    Creates a PID folder and a date subfolder inside the logged-in user's Box folder.
    Allows cutting, labeling (by task), and saving audio segments.
    """

    st.subheader("💾 Audio Saver — Segment and Label Incoming Audio")

    # --- Inputs from PyQt5 or manual entry ---
    date = st.text_input("Recording Date (YYYY-MM-DD)")
    pid = st.text_input("PID (Unique Patient/Session ID)")
    audio_file = st.file_uploader("Upload or Receive Audio File", type=["wav", "mp3"])

    if not (pid and date and audio_file):
        st.info("Please enter PID, date, and upload an audio file to continue.")
        return

    # --- Each PID folder lives inside the user's folder ---
    client = get_box_client()
    pid_folder_id = ensure_task_folder(client, folder_id, pid)

    # --- Create session subfolder for date ---
    session_folder_id = ensure_task_folder(client, pid_folder_id, date)

    # --- Read and preview audio ---
    raw = audio_file.read()
    try:
        y, sr = read_audio_bytes(raw)
    except Exception as e:
        st.error(f"Failed to read the audio file: {e}")
        return

    st.caption(f"Sample rate: {sr} Hz · Duration: {len(y)/sr:.2f} s")

    # --- Interactive waveform for trimming ---
    temp_path = save_temp_mono_wav(y, sr)
    st.info("Use the waveform to select and label segments for each task below.")
    result = audix(temp_path)
    try:
        os.unlink(temp_path)
    except Exception:
        pass

    # --- Task selection checkboxes ---
    st.markdown("### Select the task for this segment:")
    selected_task = None
    cols = st.columns(2)
    for i, task in enumerate(TASKS):
        if cols[i % 2].checkbox(task, key=f"task_{i}"):
            selected_task = task

    # --- Segment logic ---
    if result and result.get("selectedRegion"):
        start = result["selectedRegion"]["start"]
        end = result["selectedRegion"]["end"]
        st.success(f"Selected region: {start:.2f}s – {end:.2f}s")

        start_idx = int(start * sr)
        end_idx = int(end * sr)
        y_segment = y[start_idx:end_idx]

        buf = io.BytesIO()
        sf.write(buf, y_segment, sr, format="WAV")
        buf.seek(0)
        st.audio(buf, format="audio/wav")

        # --- Save Segment ---
        if st.button("💾 Save Segment to Box"):
            if not selected_task:
                st.warning("Please select a task before saving.")
            else:
                safe_task_name = selected_task.replace("/", "-").replace(":", "-")
                filename = f"{pid}_{safe_task_name}.wav"
                upload_to_user_box(client, session_folder_id, filename, buf.getvalue())
                st.success(f"✅ Segment saved successfully as {filename}")
                st.toast(f"Saved {filename} in PID/{date}/")
    else:
        st.info("Select a region in the waveform to cut and save.")
