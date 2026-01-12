import streamlit as st
import os
import parselmouth as pm
import matplotlib.pyplot as plt
import pandas as pd
from st_audiorec import st_audiorec
from streamlit_advanced_audio import audix
from analysis_utils import *
import io
from PIL import Image
import google.generativeai as genai


def init_gemini():
    """
    Gemini via AI Studio API key stored securely in Streamlit Secrets or env var.
    Prefer Streamlit Cloud Secrets: GOOGLE_API_KEY = "..."
    """
    api_key = None
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None, "Missing GOOGLE_API_KEY. Add it to Streamlit Secrets (recommended) or as an environment variable."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model, None


def fig_to_pil_image(fig):
    """
    Convert a Matplotlib figure to a PIL Image (PNG) for Gemini multimodal input.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gemini_review_voice(model, df_features: pd.DataFrame, spectrogram_fig, task_name: str, reference_group: str):
    """
    Sends features + spectrogram to Gemini for interpretation.
    Does NOT ask Gemini to infer gender/sex. Uses user-selected reference group.
    """
    rows = df_features.to_dict(orient="records")
    spec_img = fig_to_pil_image(spectrogram_fig)

    prompt = f"""
You are assisting with voice acoustics interpretation for the task: "{task_name}".

IMPORTANT:
- Do NOT infer or guess gender/sex/identity from audio or images.
- Use the selected reference group only: "{reference_group}".
- If reference group is "Unknown / show both", provide ranges/interpretation for typical adult male and typical adult female, without guessing which applies.
- Do not provide a medical diagnosis. Use cautious, non-diagnostic language.

Input data:
1) Extracted acoustic features (Feature, Value): {rows}
2) Spectrogram image attached.

Please produce:
A) Summary (2–5 sentences)
B) Range check vs reference group (bullets). If a feature is out of typical ranges, say so with uncertainty and mention it depends on recording/task.
C) Any potential flags (bullets) — only if supported by the data/spectrogram; otherwise “No obvious flags.”
D) Suggestions (bullets): e.g., repeat recording conditions, consult clinician if symptoms exist, etc.
"""
    # Multimodal: prompt + image
    resp = model.generate_content([prompt, spec_img])
    return resp.text if hasattr(resp, "text") else str(resp)



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
    st.caption("Click to record, then stop. The widget shows a waveform while recording.")
    # Recorder reloads whenever a new task is chosen
    recorder_key = st.session_state.get("recorder_reload_key", f"recorder_{selected_task}")
    wav_audio_data = st_audiorec()  # cannot take key argument; reload handled via rerun


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

                    # ---------------- GEMINI REVIEW (features + spectrogram) - ADD ONLY ----------------
                    with st.expander("Gemini review (features + spectrogram)", expanded=False):
                        model, err = init_gemini()
                        if err:
                            st.warning(err)
                        else:
                            reference_group = st.radio(
                                "Reference group for typical ranges (we won't guess it from the audio):",
                                options=["Unknown / show both", "Adult male (self-reported)", "Adult female (self-reported)"],
                                index=0,
                                horizontal=True,
                                key="gemini_reference_group",
                            )

                            if st.button("Run Gemini review", key="gemini_review_btn"):
                                with st.spinner("Sending features + spectrogram to Gemini..."):
                                    review_text = gemini_review_voice(
                                        model=model,
                                        df_features=df,
                                        spectrogram_fig=figs["spectrogram"],
                                        task_name=selected_task,
                                        reference_group=reference_group,
                                    )
                                st.markdown(review_text)
                    # -------------------------------------------------------------------------------


                    if save_auto:
                        with st.spinner("Saving the analysis", show_time=True):
                            save_analysis_to_box(y_region, sr, df, figs, folder_id)
                        st.success("Analysed and Saved results")
                    else:
                        st.info("Analysis completed (not saved). Check 'Save automatically' to reanalyse and store the results.")
