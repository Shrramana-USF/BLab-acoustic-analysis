# audio_saver_mode.py
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

def audio_saver_tab(folder_id):
    """
    Audio Saver tab (per-user).
    Receives date, PID, and an audio file.
    Creates a PID folder inside the logged-in user's Box folder.
    Allows cutting and saving audio segments.
    """

    st.subheader("Audio Saver — Segment and Save Incoming Files")

    # --- Inputs from PyQt5 or manual entry ---
    date = st.text_input("Recording Date (YYYY-MM-DD)")
    pid = st.text_input("PID (Unique Patient/Session ID)")
    audio_file = st.file_uploader("Upload or Receive Audio File", type=["wav", "mp3"])

    if not (pid and audio_file):
        st.info("Please enter PID and upload an audio file to continue.")
        return

    # --- Each PID folder lives inside this user’s folder ---
    client = get_box_client()
    pid_folder_id = ensure_task_folder(client, folder_id, pid)

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
    st.info("Use the waveform to select and save audio segments.")
    result = audix(temp_path)
    try:
        os.unlink(temp_path)
    except Exception:
        pass

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

        if st.button("Save Segment to Box"):
            filename = f"{pid}_{date}_{start:.2f}-{end:.2f}.wav"
            upload_to_user_box(client, pid_folder_id, filename, buf.getvalue())
            st.success(f"Segment saved successfully as {filename}")
    else:
        st.info("Select a region in the waveform to cut and save.")
