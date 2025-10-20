import streamlit as st
import io
import os
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import parselmouth as pm
from streamlit_advanced_audio import audix

from analysis_utils import (
    get_box_client,
    ensure_task_folder,
    upload_to_user_box,
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


# ----------------- Utility functions -----------------

def estimate_f0_praat(pitch):
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values != 0]
    if len(pitch_values) == 0:
        return None
    return np.mean(pitch_values)


def summarize_features(snd, pitch, intensity):
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values != 0]
    inten_values = intensity.values.T.flatten()
    return {
        "Pitch mean (Hz)": np.mean(pitch_values) if len(pitch_values) else np.nan,
        "Pitch std (Hz)": np.std(pitch_values) if len(pitch_values) else np.nan,
        "Pitch min (Hz)": np.min(pitch_values) if len(pitch_values) else np.nan,
        "Pitch max (Hz)": np.max(pitch_values) if len(pitch_values) else np.nan,
        "Intensity mean (dB)": np.mean(inten_values) if len(inten_values) else np.nan,
        "Intensity std (dB)": np.std(inten_values) if len(inten_values) else np.nan,
        "Duration (s)": snd.duration,
    }


def pitch_contour(pitch):
    return pitch.xs(), pitch.selected_array["frequency"]


def intensity_contour(intensity):
    return intensity.xs(), intensity.values.T.flatten()


def compute_spectrogram(snd):
    return snd.to_spectrogram(window_length=0.01, maximum_frequency=8000)


def plot_spectrogram(spectrogram):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(np.maximum(1e-10, spectrogram.values))
    fig, ax = plt.subplots()
    img = ax.pcolormesh(X, Y, sg_db, shading="auto", cmap="viridis")
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, label="Intensity (dB)")
    return fig


def fetch_file_from_box(client, folder_id, filename):
    """
    Look for a file by name in a Box folder and return its bytes.
    """
    folder_items = client.folders.get_folder_items(str(folder_id))
    file_item = next((f for f in folder_items.entries if f.type == "file" and f.name == filename), None)
    if not file_item:
        raise FileNotFoundError(f"File '{filename}' not found in folder {folder_id}")
    byte_stream = client.downloads.download_file(file_item.id)
    return read_byte_stream(byte_stream)


def report_exists_in_box(client, folder_id, date, task):
    """
    Check if a report (CSV or plots) already exists for the given date and task.
    """
    expected_names = [
        f"{date}_{task}_features.csv",
        f"{date}_{task}_pitch.png",
        f"{date}_{task}_intensity.png",
        f"{date}_{task}_spectrogram.png",
    ]
    try:
        items = client.folders.get_folder_items(str(folder_id))
        existing_names = [f.name for f in items.entries if f.type == "file"]
        return any(name in existing_names for name in expected_names)
    except Exception as e:
        st.warning(f"Could not verify report existence: {e}")
        return False


# ----------------- Main Streamlit Tab -----------------

def split_audio_report_tab(folder_id):
    """
    Split Audio Report Tab
    Loads saved audio segments from Box,
    plays them in Streamlit, extracts features and plots,
    and saves results to Box (unless report already exists).
    """

    st.subheader("Split Audio Report")

    pid = st.text_input("PID (Unique Patient/Session ID)", key="split_pid")
    date = st.text_input("Session Date (YYYY-MM-DD)", key="split_date")
    task = st.selectbox("Select Task", TASKS, key="split_task")

    if not (pid and date and task):
        st.info("Please fill in all fields to continue.")
        return

    client = get_box_client()
    pid_folder_id = ensure_task_folder(client, folder_id, pid)
    task_folder_id = ensure_task_folder(client, pid_folder_id, task)
    filename = f"{date}_{task}.wav"

    st.write(f"Loading file from Box: {pid}/{task}/{filename}")

    try:
        audio_bytes = fetch_file_from_box(client, task_folder_id, filename)
    except Exception as e:
        st.error(f"Could not find or load audio file: {e}")
        return

    # Display simple player for playback
    st.info("Loaded audio file for playback and analysis:")
    st.audio(audio_bytes, format="audio/wav")

    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        st.caption(f"Loaded {filename} — Duration: {len(y)/sr:.2f}s, SR: {sr}Hz")
    except Exception as e:
        st.error(f"Error reading audio: {e}")
        return

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

                # Feature summary
                features = summarize_features(snd, pitch, intensity)
                df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
                st.dataframe(df, width="stretch", hide_index=True)

                # Plots
                xs, f0_contour = pitch_contour(pitch)
                fig_pitch, ax = plt.subplots()
                ax.plot(xs, f0_contour, color="blue")
                ax.set_title("Pitch contour")
                st.pyplot(fig_pitch)

                xs, inten_contour = intensity_contour(intensity)
                fig_intensity, ax = plt.subplots()
                ax.plot(xs, inten_contour, color="green")
                ax.set_title("Intensity contour")
                st.pyplot(fig_intensity)

                spectrogram = compute_spectrogram(snd)
                fig_spec = plot_spectrogram(spectrogram)
                st.pyplot(fig_spec)

                # Save if report does not already exist
                if report_exists_in_box(client, task_folder_id, date, task):
                    st.warning(f"A report already exists for this session ({date} — {task}).")
                    st.info("Analysis completed but results were not saved again.")
                else:
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
            except Exception as e:
                st.error(f"Feature extraction failed: {e}")
