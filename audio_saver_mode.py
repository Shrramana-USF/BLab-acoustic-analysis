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

# Define all task labels
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
    Folder structure:
        /UserFolder/PID/TASKNAME/sessiondate_taskname.wav
    """

    st.subheader("Audio Saver — Segment and Label Incoming Audio")

    # --- Inputs ---
    pid = st.text_input("PID (Unique Patient/Session ID)")
    date = st.text_input("Session Date (YYYY-MM-DD)")
    audio_file = st.file_uploader("Upload or Receive Audio File", type=["wav", "mp3"])

    if not (pid and date and audio_file):
        st.info("Please enter PID, date, and upload an audio file to continue.")
        return

    # --- Initialize Box client ---
    client = get_box_client()

    # --- Step 1: Ensure PID folder exists ---
    pid_folder_id = ensure_task_folder(client, folder_id, pid)

    # --- Step 2: Load audio data ---
    raw = audio_file.read()
    try:
        y, sr = read_audio_bytes(raw)
    except Exception as e:
        st.error(f"Failed to read the audio file: {e}")
        return

    st.caption(f"Sample rate: {sr} Hz · Duration: {len(y)/sr:.2f} s")

    # --- Step 3: Interactive waveform ---
    temp_path = save_temp_mono_wav(y, sr)
    st.info("Use the waveform to select a region, choose a task, and save.")
    result = audix(temp_path)
    try:
        os.unlink(temp_path)
    except Exception:
        pass

    # --- Step 4: Choose task ---
    st.markdown("### Select the task for this segment:")
    selected_task = None
    cols = st.columns(2)
    for i, task in enumerate(TASKS):
        if cols[i % 2].checkbox(task, key=f"task_{i}"):
            selected_task = task

    # --- Step 5: Handle segment extraction ---
    if result and result.get("selectedRegion"):
        start = result["selectedRegion"]["start"]
        end = result["selectedRegion"]["end"]
        st.success(f"Selected region: {start:.2f}s – {end:.2f}s")

        start_idx = int(start * sr)
        end_idx = int(end * sr)
        y_segment = y[start_idx:end_idx]

        buf = io.BytesIO()
        sf.write(buf, y_segment, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        st.audio(buf, format="audio/wav")

        # --- Step 6: Save segment ---
        if st.button("Save Segment to Box"):
            if not selected_task:
                st.warning("Please select a task before saving.")
            else:
                safe_task = selected_task.replace("/", "-").replace(":", "-")

                # Ensure task folder inside PID
                task_folder_id = ensure_task_folder(client, pid_folder_id, safe_task)

                # Filename: <date>_<task>.wav
                filename = f"{date}_{safe_task}.wav"

                upload_to_user_box(client, task_folder_id, filename, buf.getvalue())
                st.success(f"Saved {filename} inside {pid}/{safe_task}/")
                st.toast(f"Saved {filename} successfully!")
    else:
        st.info("Select a region in the waveform to cut and save.")
