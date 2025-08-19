import streamlit as st

st.title("📊 Dashboard Page")

if st.button("Go to Home"):
    st.switch_page("pages.home")  # Switch back to home.py