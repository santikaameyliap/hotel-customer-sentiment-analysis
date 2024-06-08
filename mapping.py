import subprocess
import streamlit as st

# Menjalankan skrip generate_plots.py
subprocess.run(['python', 'dash.py'])

# Menampilkan grafik di aplikasi Streamlit
st.markdown(f"## Word Count Plots")
st.markdown(f"![Word Count Plots]({word-plots})")