import streamlit as st
import os, io, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parselmouth as pm
import soundfile as sf
import time
from analysis_utils import *
from streamlit_advanced_audio import audix


# ---------------- UPLOAD TAB ----------------
def upload_tab(folder_id):
    st.subheader("Upload Audio for Task")

    # --- Step 1: Task Selection ---
    tasks = [ "Rainbow passage", "Maximum sustained phonation on 'aaah'", "Comfortable sustained phonation on 'eeee'", 
             "Glide up to your highest pitch on 'eeee'", "Glide down to your lowest pitch on 'eeee'", 
             "Sustained 'aaah' at minimum volume", "Maximum loudness level (brief 'AAAH')", "Conversational speech"]
    
    selected_task = st.radio(
        "Select a task to continue:",
        options=tasks,
        index=None,
        horizontal=True
    )

    # --- Step 2: Stop here if no task chosen yet ---
    # Stop until a task is chosen
    if selected_task is None:
        st.info("Please select a task to enable uploading.")
        return

    st.markdown(f"### Selected Task: {selected_task}")

    # --- Create / get task folder ---
    client = get_box_client()
    task_folder_id = ensure_task_folder(client, folder_id, selected_task)

    # --- Task-scoped widget keys to force reset when task changes ---
    uploader_key   = f"upload_uploader_{selected_task}"
    save_auto_key  = f"upload_save_auto_{selected_task}"
    analyze_btn_key = f"upload_analyze_{selected_task}"

    # --- Uploader appears only after task is selected ---
    up = st.file_uploader(
        "Upload audio (WAV only)",
        type=["wav"],
        key=uploader_key,        # <-- new widget when task changes -> empty
    )
    if up is None:
        st.info("Upload a WAV file to begin analysis.")
        return

    # Read audio
    raw = up.read()
    try:
        y, sr = read_audio_bytes(raw)
    except Exception:
        st.error("Could not decode this WAV file.")
        return

    st.caption(f"Sample rate: {sr} Hz  ·  Duration: {len(y)/sr:.2f} s")

    # Ensure mono file for waveform widget
    temp_path = save_temp_mono_wav(y, sr)
    result = audix(temp_path)

    # cleanup temp
    try:
        os.unlink(temp_path)
    except Exception:
        pass

    st.caption("Trim the audio to analyse a selected portion")
    save_auto = st.checkbox("Save the analysis automatically", key=save_auto_key)

    y_region = None
    if st.button("Analyse Audio", key=analyze_btn_key):
        if result and result.get("selectedRegion"):
            start = result["selectedRegion"]["start"]
            end = result["selectedRegion"]["end"]
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            y_region = y[start_idx:end_idx]
            st.info(f"Analysing selected region: {start:.2f}s – {end:.2f}s")
        else:
            y_region = y
            st.info("No region selected: Analysing entire file")

    if y_region is None or len(y_region) == 0:
        return

    # Analysis (unchanged)
    snd = pm.Sound(y_region, sampling_frequency=sr)
    pitch = pm.praat.call(snd, "To Pitch (filtered autocorrelation)", 0.0, 30.0, 600.0, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14)
    intensity = snd.to_intensity()

    f0 = estimate_f0_praat(pitch)
    if f0 is None:
        st.warning("No stable fundamental frequency detected.")
        return

    features = summarize_features(snd, pitch, intensity)
    df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
    st.dataframe(df, width="stretch", hide_index=True)

    figs = {}

    xs, f0_contour = pitch_contour(pitch)
    fig, ax = plt.subplots()
    ax.plot(xs, f0_contour, color="blue")
    ax.set_title("Pitch contour")
    st.pyplot(fig)
    figs["pitch"] = fig

    xs, inten_contour = intensity_contour(intensity)
    fig, ax = plt.subplots()
    ax.plot(xs, inten_contour, color="green")
    ax.set_title("Intensity contour")
    st.pyplot(fig)
    figs["intensity"] = fig

    spectrogram = compute_spectrogram(snd)
    fig = plot_spectrogram(spectrogram)
    st.pyplot(fig)
    figs["spectrogram"] = fig

    if save_auto:
        with st.spinner("Saving the analysis", show_time=True):
            # save under user/<task>/session_...
            save_analysis_to_box(y_region, sr, df, figs, task_folder_id)
        st.success("Analysed and saved results.")
        st.toast("Analysed and saved results.")
    else:
        st.info("Analysis completed (not saved). Check 'Save automatically' to reanalyse and store results.")
        st.toast("Analysis completed (not saved).")