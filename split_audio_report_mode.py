import streamlit as st
import io
import os
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import parselmouth as pm  # for Praat-like analysis
from streamlit_advanced_audio import audix

from analysis_utils import (
    get_box_client,
    ensure_task_folder,
    upload_to_user_box,
)

# --- Task List ---
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


# --- Analysis Utility Functions ---
def estimate_f0_praat(pitch):
    """Estimate mean fundamental frequency (F0) from a Parselmouth Pitch object."""
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]
    if len(pitch_values) == 0:
        return None
    return np.mean(pitch_values)


def summarize_features(snd, pitch, intensity):
    """Compute summary statistics for pitch and intensity."""
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]
    inten_values = intensity.values.T.flatten()
    features = {
        "Pitch mean (Hz)": np.mean(pitch_values) if len(pitch_values) else np.nan,
        "Pitch std (Hz)": np.std(pitch_values) if len(pitch_values) else np.nan,
        "Pitch min (Hz)": np.min(pitch_values) if len(pitch_values) else np.nan,
        "Pitch max (Hz)": np.max(pitch_values) if len(pitch_values) else np.nan,
        "Intensity mean (dB)": np.mean(inten_values) if len(inten_values) else np.nan,
        "Intensity std (dB)": np.std(inten_values) if len(inten_values) else np.nan,
        "Duration (s)": snd.duration,
    }
    return features


def pitch_contour(pitch):
    xs = pitch.xs()
    ys = pitch.selected_array['frequency']
    return xs, ys


def intensity_contour(intensity):
    xs = intensity.xs()
    ys = intensity.values.T.flatten()
    return xs, ys


def compute_spectrogram(snd):
    return snd.to_spectrogram(window_length=0.01, maximum_frequency=8000)


def plot_spectrogram(spectrogram):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(np.maximum(1e-10, spectrogram.values))
    fig, ax = plt.subplots()
    img = ax.pcolormesh(X, Y, sg_db, shading='auto', cmap='viridis')
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, label="Intensity (dB)")
    return fig


def download_file_from_box(client, folder_id, filename):
    """Fetch file bytes from Box by name (used to feed into audix and analysis)."""
    folder_items = client.folder(folder_id=folder_id).get_items(limit=1000)
    for item in folder_items:
        if item.name == filename:
            return client.file(item.id).content()
    raise FileNotFoundError(f"File '{filename}' not found in folder ID {folder_id}")


# --- Streamlit Tab ---
def split_audio_report_tab(folder_id):
    """
    Split Audio Report Tab
    - Loads saved audio segments from Box
    - Displays waveform using advanced audio widget
    - Extracts and saves audio features + plots to Box
    """

    st.subheader("Split Audio Report — Analyze Audio Segments")

    pid = st.text_input("PID (Unique Patient/Session ID)")
    date = st.text_input("Session Date (YYYY-MM-DD)")
    task = st.selectbox("Select Task", TASKS)

    if not (pid and date and task):
        st.info("Please fill in all fields to continue.")
        return

    client = get_box_client()

    # --- Folder structure ---
    pid_folder_id = ensure_task_folder(client, folder_id, pid)
    task_folder_id = ensure_task_folder(client, pid_folder_id, task)
    filename = f"{date}_{task}.wav"

    st.write(f"Loading file from Box: **{pid}/{task}/{filename}**")

    try:
        audio_bytes = download_file_from_box(client, task_folder_id, filename)
    except Exception as e:
        st.error(f"Could not find or load audio file: {e}")
        return

    # --- Display waveform using advanced audio widget ---
    with st.expander("View and inspect the waveform"):
        st.info("Use the waveform below to listen or inspect before analysis.")
        # Save temp file to feed into audix
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        result = audix(temp_path)

    # --- Load into numpy for analysis ---
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        st.caption(f"Loaded {filename} — Duration: {len(y)/sr:.2f}s, SR: {sr}Hz")
    except Exception as e:
        st.error(f"Error reading audio: {e}")
        return

    # --- Extract features ---
    if st.button("Extract and Save Features"):
        with st.spinner("Analyzing audio... please wait"):
            try:
                snd = pm.Sound(y, sampling_frequency=sr)
                pitch = snd.to_pitch(time_step=None, pitch_floor=10, pitch_ceiling=5000)
                intensity = snd.to_intensity()

                f0 = estimate_f0_praat(pitch)
                if f0 is None:
                    st.warning("No stable fundamental frequency detected.")
                    return

                features = summarize_features(snd, pitch, intensity)
                df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
                st.dataframe(df, width="stretch", hide_index=True)

                # --- Plot Pitch ---
                xs, f0_contour = pitch_contour(pitch)
                fig_pitch, ax = plt.subplots()
                ax.plot(xs, f0_contour, color="blue")
                ax.set_title("Pitch contour")
                st.pyplot(fig_pitch)

                # --- Plot Intensity ---
                xs, inten_contour = intensity_contour(intensity)
                fig_intensity, ax = plt.subplots()
                ax.plot(xs, inten_contour, color="green")
                ax.set_title("Intensity contour")
                st.pyplot(fig_intensity)

                # --- Plot Spectrogram ---
                spectrogram = compute_spectrogram(snd)
                fig_spec = plot_spectrogram(spectrogram)
                st.pyplot(fig_spec)

                # --- Save all outputs to Box ---
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                upload_to_user_box(client, task_folder_id, f"{date}_{task}_features.csv", csv_buf.getvalue().encode("utf-8"))

                img_buf = io.BytesIO()
                fig_pitch.savefig(img_buf, format="png"); img_buf.seek(0)
                upload_to_user_box(client, task_folder_id, f"{date}_{task}_pitch.png", img_buf.getvalue())

                img_buf = io.BytesIO()
                fig_intensity.savefig(img_buf, format="png"); img_buf.seek(0)
                upload_to_user_box(client, task_folder_id, f"{date}_{task}_intensity.png", img_buf.getvalue())

                img_buf = io.BytesIO()
                fig_spec.savefig(img_buf, format="png"); img_buf.seek(0)
                upload_to_user_box(client, task_folder_id, f"{date}_{task}_spectrogram.png", img_buf.getvalue())

                st.success("Features and plots saved to Box successfully.")
                st.toast(f"Analysis for {task} ({date}) saved.")
            except Exception as e:
                st.error(f"Feature extraction failed: {e}")
