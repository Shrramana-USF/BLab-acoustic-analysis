import streamlit as st


# Import tabs (modes)
from upload_mode import upload_tab
from record_mode import record_tab
from report_mode import report_tab
from audio_saver_mode import audio_saver_tab
from split_audio_report_mode import split_audio_report_tab
from split_audio_trend_mode import split_audio_trend_tab



# Import utilities
from analysis_utils import *

# ---------------- Global Config ----------------
st.set_page_config(page_title="BLab Anaysis Tool")
st.title("BLab Acoustic Analysis Dashboard")
st.caption("Backend processing with PRAAT")




# ---------------- AUTHENTICATION ----------------
if not st.user.is_logged_in:
    if st.button("Log in with Google"):
        st.login()
    st.stop()

username = st.user.name
email = st.user.email
folder_id, is_new = handle_user_login(username, email)

if is_new:
    st.success(f"New profile created for {username}")
else:
    st.info(f"Welcome back {username}")

if st.button("Log out"):
    st.logout()

st.warning("DO NOT FORGET TO LOGOUT!")

# ---------------- MODE TABS ----------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload",
    "🎧 Record",
    "📊 Report",
    "💾 Audio Saver",
    "📈 Split Audio Report",
    "📉 Split Audio Trend"
])

with tab1:
    upload_tab(folder_id)
with tab2:
    record_tab(folder_id)
with tab3:
    report_tab(folder_id)
with tab4:
    audio_saver_tab(folder_id)
with tab5:
    split_audio_report_tab(folder_id)
with tab6:
    split_audio_trend_tab(folder_id)

