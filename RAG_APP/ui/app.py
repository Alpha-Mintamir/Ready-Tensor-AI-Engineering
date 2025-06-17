import streamlit as st
import json
import os
import sys
from datetime import datetime

# Now use a regular import
from processing.embeddings import query_db
from core.services import generate_answer, process_and_index_files, upload_file,save_chat_history, load_chat_history
# --- Helper Functions ---

# --- Session State Initialization ---

if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(st.session_state.session_id)
if "threads" not in st.session_state:
    st.session_state.threads = {}

# --- Sidebar: Branding and Thread Management ---

st.sidebar.image("https://banner2.cleanpng.com/20180411/toq/kisspng-history-television-channel-logo-story-5acdebba618e64.3176815315234446663996.jpg", width=120)  # Place your logo at ui/logo.png
st.sidebar.title("RAG Chatbot")
st.sidebar.markdown("**Chat about U.S. history.**\n\nUpload documents, ask questions, and get answers with sources.")

thread_names = list(st.session_state.threads.keys())
selected_thread = st.sidebar.selectbox("Select thread", ["New thread"] + thread_names)

if selected_thread == "New thread":
    new_thread_name = st.sidebar.text_input("Thread name")
    if st.sidebar.button("Create thread") and new_thread_name:
        st.session_state.threads[new_thread_name] = []
        st.session_state.session_id = new_thread_name
        st.session_state.chat_history = []
        save_chat_history(new_thread_name, [])
        st.rerun()
else:
    st.session_state.session_id = selected_thread
    st.session_state.chat_history = load_chat_history(selected_thread)

# --- Main UI ---

st.markdown(
    """
    <div style='display: flex; align-items: center;'>
        <img src='https://img.freepik.com/free-vector/chatbot-chat-message-vectorart_78370-4104.jpg?ga=GA1.1.1769002499.1718555265&semt=ais_items_boosted&w=740' width='60' style='margin-right: 15px;'>
        <h1 style='margin-bottom: 0;'>RAG Chatbot: U.S. History</h1>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# File Upload
with st.expander("📄 Upload Documents", expanded=False):
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        st.info("Uploading and processing files...")
        upload_files = [upload_file(file) for file in uploaded_files]
        process_and_index_files(uploaded_files, st.session_state.session_id)
        st.success("Files uploaded and processed!")
        st.session_state.updating_kb = False

# Chat Interface
st.subheader("💬 Chat")
user_input = st.text_input("Type your question...", key="user_input")
if st.button("Send") and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Retrieving answer..."):
        response = generate_answer(user_input)
    st.session_state.chat_history.append({"role": "bot", "content": response["answer"], "sources": response["sources"]})
    st.session_state.threads[st.session_state.session_id] = st.session_state.chat_history
    save_chat_history(st.session_state.session_id, st.session_state.chat_history)
    st.rerun()

# Display Chat History
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div style='background:#e6f7ff;padding:8px;border-radius:8px;margin-bottom:4px'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#f6ffed;padding:8px;border-radius:8px;margin-bottom:4px'><b>Bot:</b> {msg['content']}<br><span style='font-size:0.9em;color:#888'>Sources: {', '.join(msg.get('sources', []))}</span></div>", unsafe_allow_html=True)

# Status Indicator
if st.session_state.get("updating_kb", False):
    st.info("Updating knowledge base...")

#footer
st.markdown(
    "<hr><center><small>Developed by Anish, Alpha, and Ogie &nbsp;|&nbsp; Powered by Streamlit & LangChain with Gemini AI &copy; 2025</small></center>",
    unsafe_allow_html=True
)