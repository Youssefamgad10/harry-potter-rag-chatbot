import streamlit as st
from rag_core import build_pipeline, ask_question

st.set_page_config(page_title="Ask Harry Potter Book 1", page_icon=":zap:")

@st.cache_resource
def load_pipeline():
    print("Starting pipeline build...", flush=True)
    pipeline = build_pipeline()
    print("Pipeline build complete.", flush=True)
    return pipeline

try:
    with st.spinner("Loading book and building search index (first load takes ~30-60 seconds)..."):
        pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to build pipeline: {e}")
    st.stop()

st.title(":zap: Ask Harry Potter: The Sorcerer's Stone")
st.caption("RAG-powered Q&A grounded in the actual book text")

question = st.text_input("Your question:", placeholder="What does Harry's uncle do for work?")

if question:
    with st.spinner("Searching the book..."):
        answer, context_chunks = ask_question(pipeline, question)
    st.markdown("### Answer")
    st.write(answer)
    with st.expander("Show retrieved excerpts"):
        for c in context_chunks:
            st.markdown(f"**Page {c['page']}** (score: {c['score']:.3f})")
            st.write(c['chunk'])
            st.divider()