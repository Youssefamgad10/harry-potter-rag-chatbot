import streamlit as st
from rag_core import build_pipeline, build_pipeline_from_pdf, ask_question
import tempfile

st.set_page_config(page_title="Ask Any Book", page_icon=":books:")

st.title(":books: Ask Any Book")
st.caption("Upload a PDF and chat with it, or use the default Harry Potter book")

with st.sidebar:
    st.header("Book Source")
    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")
    use_default = st.button("Use default book (Harry Potter)")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.book_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if use_default:
    with st.spinner("Loading default book..."):
        st.session_state.pipeline = build_pipeline()
        st.session_state.book_name = "Harry Potter and the Sorcerer's Stone (default)"
        st.session_state.messages = []
    st.rerun()

if uploaded_file is not None and st.session_state.book_name != uploaded_file.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        with st.spinner(f"Reading and indexing '{uploaded_file.name}' (this may take a minute)..."):
            st.session_state.pipeline = build_pipeline_from_pdf(tmp_path)
            st.session_state.book_name = uploaded_file.name
            st.session_state.messages = []
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        st.stop()

if st.session_state.pipeline is None:
    st.info("Upload a PDF in the sidebar, or click 'Use default book' to get started.")
    st.stop()

st.success(f"Currently chatting with: **{st.session_state.book_name}**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "context" in message:
            with st.expander("Show retrieved excerpts"):
                for c in message["context"]:
                    page = c["page"]
                    score = c["score"]
                    st.markdown(f"**Page {page}** (score: {score:.3f})")
                    st.write(c["chunk"])
                    st.divider()

if question := st.chat_input("Ask a question about the book..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the book..."):
            answer, context_chunks = ask_question(st.session_state.pipeline, question)
        st.write(answer)
        with st.expander("Show retrieved excerpts"):
            for c in context_chunks:
                page = c["page"]
                score = c["score"]
                st.markdown(f"**Page {page}** (score: {score:.3f})")
                st.write(c["chunk"])
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context_chunks
    })