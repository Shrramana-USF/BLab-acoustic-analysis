import streamlit as st
import base64
import io

# ---------------------------------------
# RECORD TAB
# ---------------------------------------

def record_tab(folder_id):
    st.header("🎙️ Record Mode")

    # --- OLD BROKEN COMPONENT (Commented Out) ---
    # from streamlit.components.v1 import declare_component
    # _media_recorder_component = declare_component(
    #     "media_recorder"
    # )

    # --- NEW INLINE IMPLEMENTATION USING MediaRecorder API ---
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

    # --- Streamlit-side listener for recorded data ---
    # (Requires the "st.experimental_get_query_params" workaround or custom JS listener)
    st.info("Press Start/Stop to record. Browser auto-gain is disabled.")
