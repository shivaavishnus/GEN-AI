import streamlit as st
import uuid
from rag_utils import (
    load_and_split_documents,
    create_chroma_db,
    create_qa_chain,
    check_cache,
    save_cache,
    get_chat_history,
    save_chat_history
)

st.set_page_config(page_title="Chat with Documents", layout="wide")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = get_chat_history(st.session_state.session_id)

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# Layout: sidebar | chat area | file upload
sidebar, main_area, file_area = st.columns([1.3, 3, 1.5])

# Sidebar: Chat history + new chat
with sidebar:
    st.title("💬 History")
    if st.button("🆕 New Chat"):
        st.session_state.chat_history = []
        st.session_state.retriever = None
        st.session_state.file_uploaded = False
        save_chat_history(st.session_state.session_id, [])
        st.query_params["chat"] = str(uuid.uuid4())  # Updated here

    for idx, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{idx+1}:** {chat['question'][:40]}...")

# Right: Upload section
with file_area:
    st.title("📂 Upload")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "cs"], accept_multiple_files=True)
    if uploaded_files and not st.session_state.file_uploaded:
        with st.spinner("Processing..."):
            docs = load_and_split_documents(uploaded_files)
            retriever = create_chroma_db(docs)
            st.session_state.retriever = retriever
            st.session_state.file_uploaded = True
        st.success("✅ Document indexed!")

# Center Chat UI
with main_area:
    st.title("🤖 Chat with your documents")

    # Show chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])

    # Bottom input area
    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question", key="chat_input", label_visibility="collapsed")
        submitted = st.form_submit_button("➤")

    if submitted and user_input:
        if not st.session_state.retriever:
            st.error("Please upload files before asking questions.")
        else:
            with st.spinner("Thinking..."):
                cached = check_cache(user_input)
                if cached:
                    answer = cached
                else:
                    chain = create_qa_chain(st.session_state.retriever)
                    answer = chain.run(user_input)
                    save_cache(user_input, answer)

                st.session_state.chat_history.append({"question": user_input, "answer": answer})
                save_chat_history(st.session_state.session_id, st.session_state.chat_history)
                st.rerun()
