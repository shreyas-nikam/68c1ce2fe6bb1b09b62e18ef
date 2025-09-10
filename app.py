
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we delve into the fascinating world of Transformer models, specifically focusing on the self-attention mechanism. This interactive application allows you to explore how Transformers understand and process sequences by weighing the importance of different words in a sentence. We will visualize attention weights, understand how synthetic data is generated and validated, and ultimately gain a deeper insight into the "Attention Is All You Need" paradigm.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Self-Attention Visualization", "Multi-Head Attention", "Positional Encoding"])
if page == "Self-Attention Visualization":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Multi-Head Attention":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Positional Encoding":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
