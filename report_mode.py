import streamlit as st
import pandas as pd
from analysis_utils import *
import matplotlib.pyplot as plt

# --- ADD: Gemini imports ---
import os
import google.generativeai as genai


# --- ADD: Gemini init helper (secure secrets/env) ---
def init_gemini():
    """
    Gemini via AI Studio API key stored securely in Streamlit Secrets or env var.
    Supports either:
      GOOGLE_API_KEY = "..."
    or:
      [Gemini]
      GOOGLE_API_KEY = "..."
    """
    api_key = None
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", None)
        if not api_key:
            api_key = st.secrets.get("Gemini", {}).get("GOOGLE_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None, "Missing GOOGLE_API_KEY in Streamlit Secrets (recommended) or environment variables."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model, None


# --- ADD: Build compact summary to send to Gemini ---
def build_trend_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    df columns expected at least: Feature, session, Value (numeric)
    Returns per-feature summary: first/last/change/mean/std/min/max/count
    """
    out_rows = []
    for feat in df["Feature"].unique():
        sub = df[df["Feature"] == feat].sort_values("session")
        vals = pd.to_numeric(sub["Value"], errors="coerce").dropna()
        if vals.empty:
            continue
        first = float(vals.iloc[0])
        last = float(vals.iloc[-1])
        out_rows.append({
            "Feature": feat,
            "count": int(vals.shape[0]),
            "first": first,
            "last": last,
            "change(last-first)": float(last - first),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    return pd.DataFrame(out_rows)


# --- ADD: Gemini report analysis ---
def gemini_analyse_report(model, selected_task: str, pivot: pd.DataFrame, trend_summary: pd.DataFrame) -> str:
    """
    Sends pivot + summary to Gemini and returns the response text.
    """
    # Keep payload reasonable
    pivot_csv = pivot.reset_index().to_csv(index=False)
    summary_csv = trend_summary.to_csv(index=False)

    prompt = f"""
You are reviewing longitudinal voice-feature results for the task: "{selected_task}".

You are given:
1) A pivot table (rows=Feature, columns=session, values=Value).
2) A trend summary per feature (first, last, change, mean, std, min, max, count).

Goals:
- Identify notable trends across sessions (improving/worsening/stable).
- Call out potential anomalies or outliers (sudden jumps, inconsistent values, missingness patterns).
- Provide non-diagnostic, cautious interpretation and practical suggestions for next steps.
- Do NOT provide a medical diagnosis.
- Do NOT infer sensitive attributes (gender/sex/identity) from the data.

Please output:
A) Executive summary (3–6 bullets)
B) Notable trends (group by feature families if obvious: pitch/intensity/perturbation/etc.)
C) Possible anomalies / outliers (bullets; cite which session/feature if possible)
D) Data quality checks (recording consistency, missing values, session count adequacy)
E) Suggestions (repeat recordings, standardize environment, consider clinician if symptoms exist)

Pivot CSV:
{pivot_csv}

Trend summary CSV:
{summary_csv}
"""
    resp = model.generate_content(prompt)
    return resp.text if hasattr(resp, "text") else str(resp)


def report_tab(folder_id):
    # --- Task selection ---
    st.subheader("Report and Analysis by Task")

    tasks = [ "Rainbow passage", "Maximum sustained phonation on 'aaah'", "Comfortable sustained phonation on 'eeee'",
             "Glide up to your highest pitch on 'eeee'", "Glide down to your lowest pitch on 'eeee'",
             "Sustained 'aaah' at minimum volume", "Maximum loudness level (brief 'AAAH')", "Conversational speech"]

    selected_task = st.radio(
        "Select a task to view reports:",
        options=tasks,
        index=None,
        horizontal=True,
        key="report_task_radio"
    )

    # --- Reset state when task changes ---
    if "prev_task_report" not in st.session_state:
        st.session_state.prev_task_report = None

    # --- ADD: persist Gemini outputs across reruns ---
    if "report_ai_text" not in st.session_state:
        st.session_state.report_ai_text = None
    if "report_ai_task" not in st.session_state:
        st.session_state.report_ai_task = None

    if selected_task != st.session_state.prev_task_report:
        st.session_state.prev_task_report = selected_task
        # Clear AI output when switching tasks
        st.session_state.report_ai_text = None
        st.session_state.report_ai_task = selected_task
        st.rerun()

    # --- Stop until a task is selected ---
    if selected_task is None:
        st.info("Please select a task to generate the report.")
        return

    # --- Check if task folder exists ---
    client = get_box_client()
    user_folder_id = str(folder_id)

    # Try to locate the matching task folder
    try:
        user_items = client.folders.get_folder_items(user_folder_id)
        task_folder = next(
            (f for f in user_items.entries if f.type == "folder" and f.name == selected_task),
            None
        )
    except Exception:
        st.error("Unable to access user folder.")
        return

    if not task_folder:
        st.warning("No sessions found for this task.")
        return

    task_folder_id = task_folder.id

    # --- Generate report only after task is selected ---
    st.info("Click 'Generate report' to analyse session trends for this task.")
    load_report = st.button("Generate Report", key=f"generate_report_{selected_task}")
    if not load_report:
        # If user already ran AI previously, still show it
        if st.session_state.report_ai_text and st.session_state.report_ai_task == selected_task:
            st.subheader("Gemini Response (previous run)")
            st.markdown(st.session_state.report_ai_text)
        return

    # ---------------- REPORT MODE ----------------
    df, audio_map = fetch_all_features(client, str(task_folder_id))

    if df.empty:
        st.warning("No features found. Run some analyses first.")
        return

    st.subheader(f"Feature analysis across sessions for {selected_task}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    pivot = df.pivot_table(index="Feature", columns="session", values="Value", aggfunc="first")
    st.dataframe(pivot, use_container_width=True)

    # --- ADD: AI button + Gemini output area ---
    st.markdown("### AI Analysis (Gemini)")
    ai_col1, ai_col2 = st.columns([1, 2])

    analyse_ai = ai_col1.button("Analyse Report with AI", key=f"analyse_report_ai_{selected_task}")
    ai_col2.caption("Uses the pivot table + trend statistics to highlight trends, outliers, and data-quality checks.")

    if analyse_ai:
        model, err = init_gemini()
        if err:
            st.session_state.report_ai_text = f"⚠️ {err}"
        else:
            trend_summary = build_trend_summary(df)
            try:
                with st.spinner("Sending report summary to Gemini..."):
                    st.session_state.report_ai_text = gemini_analyse_report(
                        model=model,
                        selected_task=selected_task,
                        pivot=pivot,
                        trend_summary=trend_summary
                    )
            except Exception as e:
                st.session_state.report_ai_text = f"Gemini failed: {e}"

    if st.session_state.report_ai_text:
        st.subheader("Gemini Response")
        st.markdown(st.session_state.report_ai_text)

    # --- Trends charts (your existing logic) ---
    features = df["Feature"].unique()
    for feat in features:
        feat_df = df[df["Feature"] == feat].sort_values("session")
        if not feat_df.empty:
            st.caption(f"Trend for {feat}")
            st.line_chart(feat_df.set_index("session")["Value"], use_container_width=True)

    # --- Sidebar audio playback ---
    st.sidebar.subheader(f"Session Audio — {selected_task}")
    for session_name in df["session"].unique():
        st.sidebar.markdown(f"**{session_name}**")
        audio_file_id = audio_map.get(session_name)
        if audio_file_id:
            byte_stream = client.downloads.download_file(audio_file_id)
            wav_bytes = read_byte_stream(byte_stream)
            st.sidebar.audio(wav_bytes, format="audio/wav")
        else:
            st.sidebar.warning("No audio found for this session")
