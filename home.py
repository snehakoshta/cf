import streamlit as st

st.title("🏡 Home Page")

if st.button("Go to Dashboard"):
    st.switch_page("pages.dashboard")  # Switch to dashboard.py