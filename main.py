from streamlit_extras.switch_page_button import switch_page
import streamlit as st
from PIL import Image
img = Image.open("2.png")
st.image(img)
col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    want_to_contribute = st.button("Start")
if want_to_contribute:
    switch_page("streamlit_app")