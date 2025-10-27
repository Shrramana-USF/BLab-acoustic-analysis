import streamlit as st
import base64
import numpy as np
import soundfile as sf
import io

# Streamlit JS bridge
from streamlit_js_eval import streamlit_js_eval


def record_tab(folder_id):
    st.header("🎙️ Record Mode")

    st.markdown("""
    <style>
    button { margin: 0.3rem 0.5rem; padding: 0.6rem 1rem; font-size: 1rem; }
    audio { margin-top: 1rem; width: 100%; }
    </style>
    <h4>Browser Recorder (No Gain Control)</h4>
    <button id="start">Start Recording</button>
    <button id="stop" disabled>Stop Recording</button>
    <audio id="playback" controls></audio>

    <script>
    let recorder, audioChunks = [];

    document.getElementById("start").onclick = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            },
            video: false
        });

        recorder = new MediaRecorder(stream);
        audioChunks = [];

        recorder.ondataavailable = e => audioChunks.push(e.data);
        recorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const base64Audio = await audioBlobToBase64(audioBlob);
            window.parent.postMessage({ type: 'AUDIO_RECORDED', audio: base64Audio }, '*');
            document.getElementById("playback").src = URL.createObjectURL(audioBlob);
        };

        recorder.start();
        document.getElementById("start").disabled = true;
        document.getElementById("stop").disabled = false;
    };

    document.getElementById("stop").onclick = () => {
        recorder.stop();
        document.getElementById("start").disabled = false;
        document.getElementById("stop").disabled = true;
    };

    async function audioBlobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }
    </script>
    """, unsafe_allow_html=True)

    st.info("Press Start/Stop to record. Browser auto-gain is disabled.")

    # --- Listen for JS postMessage (AUDIO_RECORDED) ---
    result = streamlit_js_eval(
        js_expressions="new Promise(resolve => {window.addEventListener('message', e => { if (e.data && e.data.type === 'AUDIO_RECORDED') resolve(e.data.audio); });});",
        key="media_recorder_listener"
    )

    if result:
        st.success("Audio received from browser 🎧")
        audio_bytes = base64.b64decode(result)

        # Display audio player
        st.audio(audio_bytes, format="audio/webm")

        # Convert to waveform for verification
        try:
            data, samplerate = sf.read(io.BytesIO(audio_bytes))
            st.write(f"Sample rate: {samplerate} Hz, Duration: {len(data)/samplerate:.2f}s")
            st.line_chart(data[: min(len(data), 20000)])  # plot first 20k samples
        except Exception as e:
            st.error(f"Could not parse audio: {e}")
