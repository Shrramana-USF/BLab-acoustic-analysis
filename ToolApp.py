import streamlit as st
import pandas as pd
import io
from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth
from box_sdk_gen.managers.uploads import UploadFileAttributes, UploadFileAttributesParentField, UploadFileVersionAttributes
from box_sdk_gen.internal.utils import read_byte_stream

# Import tabs (modes)
from upload_mode import upload_tab
from record_mode import record_tab
from report_mode import report_tab

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
tab1, tab2, tab3 = st.tabs(["📤 Upload", "🎧 Record", "📊 Report"])

with tab1:
    upload_tab(folder_id)

with tab2:
    record_tab(folder_id)

with tab3:
    report_tab(folder_id)
