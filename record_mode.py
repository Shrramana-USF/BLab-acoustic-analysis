import streamlit as st
import os
import parselmouth as pm
import matplotlib.pyplot as plt
import pandas as pd
# from st_audiorec import st_audiorec  # OLD recorder (commented)
from streamlit_advanced_audio import audix
from analysis_utils import *
import tempfile
import io
import base64
import numpy as np
from streamlit.components.v1 import declare_component
import soundfile as sf


# --- NEW: Define the MediaStream-based recorder Streamlit component ---
_media_recorder_component = declare_component(
    "media_recorder"
)


def media_recorder(key="media_recorder"):
    """
    Pure MediaStream + MediaRecorder based audio capture for Streamlit.
    Disables AGC, echo cancellation, and noise suppression.
    Returns raw WAV bytes (base64 decoded) once recording is stopped.
    """
    wav_b64 = _media_recorder_component(key=key, default=None)
    if wav_b64 is None:
        return None

    try:
        return base64.b64decode(wav_b64)
    except Exception:
        st.error("Error decoding recorded audio data.")
        return None


# --- HTML + JS frontend injected dynamically ---
def inject_media_recorder_frontend():
    """
    Injects MediaStream-based recorder UI into Streamlit directly.
    This version disables all browser-level processing (AGC, suppression).
    """
    st.markdown(
        """
        <div style="margin-bottom:0.5em;">
            <button id="startRec">🎙️ Start Recording</button>
            <button id="stopRec" disabled>⏹️ Stop</button>
            <span id="status">Idle</span>
        </div>

        <script>
        let recorder, audioChunks = [];
        const startBtn = document.getElementById('startRec');
        const stopBtn = document.getElementById('stopRec');
        const status = document.getElementById('status');

        startBtn.onclick = async () => {
            try {
                const constraints = {
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false,
                        channelCount: 1,
                        sampleRate: 44100
                    }
                };

                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                
                // --- Verify actual applied settings ---
                const track = stream.getAudioTracks()[0];
                const settings = track.getSettings();
                console.log('Applied audio settings:', settings);

                recorder = new MediaRecorder(stream);
                audioChunks = [];

                recorder.ondataavailable = e => {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };

                recorder.onstop = async () => {
                    const blob = new Blob(audioChunks, { type: 'audio/wav' });
                    const arrayBuffer = await blob.arrayBuffer();
                    const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

                    const msg = {
                        isStreamlitMessage: true,
                        type: "streamlit:setComponentValue",
                        value: base64Audio
                    };
                    window.parent.postMessage(msg, "*");

                    status.innerText = "✅ Recording complete!";
                    startBtn.disabled = false;
                    stopBtn.disabled = true;

                    // Stop tracks to release microphone
                    track.stop();
                };

                recorder.start();
                status.innerText = "🎙️ Recording...";
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } catch (err) {
                alert("Microphone access denied or unavailable: " + err);
            }
        };

        stopBtn.onclick = () => {
            if (recorder && recorder.state !== "inactive") {
                recorder.stop();
                status.innerText = "Processing...";
            }
        };
        </script>
        """,
        unsafe_allow_html=True,
    )


def record_tab(folder_id):
    # --- Task selection (added) ---
    st.subheader("Record Audio for Task")
    tasks = [ "Rainbow passage", "Maximum sustained phonation on 'aaah'", "Comfortable sustained phonation on 'eeee'", 
             "Glide up to your highest pitch on 'eeee'", "Glide down to your lowest pitch on 'eeee'", 
             "Sustained 'aaah' at minimum volume", "Maximum loudness level (brief 'AAAH')", "Conversational speech"]
    selected_task = st.radio(
        "Select a task to continue:",
        options=tasks,
        index=None,  # none pre-selected
        horizontal=True,
        key="record_task_radio" 
    )

    # --- Reset UI and reload recorder when switching tasks ---
    if "prev_task_record" not in st.session_state:
        st.session_state.prev_task_record = None
    if selected_task != st.session_state.prev_task_record:
        st.session_state.prev_task_record = selected_task
        st.session_state.recorder_reload_key = f"recorder_{selected_task}"
        st.rerun()

    if selected_task is None:
        st.info("Please select a task to enable recording.")
        return

    # --- Ensure Box subfolder for selected task ---
    client = get_box_client()
    folder_id = ensure_task_folder(client, folder_id, selected_task)

    # ---------------- RECORD MODE ----------------
    st.caption("Click to record, then stop. This uses the MediaStream API (no gain control, no suppression).")
    recorder_key = st.session_state.get("recorder_reload_key", f"recorder_{selected_task}")

    # --- OLD CODE (COMMENTED OUT) ---
    # wav_audio_data = st_audiorec()  # cannot take key argument; reload handled via rerun

    # --- NEW CODE: MediaStream Recorder ---
    inject_media_recorder_frontend()
    wav_audio_data = media_recorder(key=recorder_key)

    if wav_audio_data is not None:
        try:
            y, sr = read_audio_bytes(wav_audio_data)
        except Exception:
            st.error("Couldn't parse recorded WAV. Try again.")
            y, sr = None, None

        if y is not None:
            st.caption(f"Sample rate: {sr} Hz  ·  Duration: {len(y)/sr:.2f} s")

            temp_path = save_temp_mono_wav(y, sr)
            result = audix(temp_path)

            try:
                os.unlink(temp_path)
            except Exception:
                pass

            st.caption("Trim the audio to analyse a selected portion")
            save_auto = st.checkbox("Save the analysis automatically", key="record_save_auto")

            y_region = None

            if st.button("Analyse Audio", key="record_analyze"):
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

            if y_region is not None and len(y_region) > 0:
                snd = pm.Sound(y_region, sampling_frequency=sr)
                pitch = snd.to_pitch(time_step=None,pitch_floor=30,pitch_ceiling=600)
                intensity = snd.to_intensity()

                f0 = estimate_f0_praat(pitch)
                if f0 is None:
                    st.warning("No stable fundamental frequency detected.")
                else:
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
                            save_analysis_to_box(y_region, sr, df, figs, folder_id)
                        st.success("Analysed and Saved results")
                    else:
                        st.info("Analysis completed (not saved). Check 'Save automatically' to reanalyse and store the results.")
