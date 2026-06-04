import streamlit as st

st.set_page_config(
    page_title="Healthcare Claims Intelligence Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_pages = """
    <style>
    div[data-testid="stSidebarNav"] ul li:first-child {
        display: none !important;
    }
    </style>
"""
st.markdown(hide_pages, unsafe_allow_html=True)
st.switch_page("pages/1_Home.py")