import streamlit as st
import tensorflow as tf

st.title("My First ML App 🚀")

st.write("TensorFlow version:", tf.__version__)

name = st.text_input("Enter your name")

if name:
    st.success(f"Hello {name} 👋")