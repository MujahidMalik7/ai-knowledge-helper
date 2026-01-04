import streamlit as st
from rag_backend import (
    extract_text, chunk_text, embed_chunks,
    build_faiss_index, retrieve_and_rerank, generate_answer_ollama,
    download_chat_history, document_summary
)

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 RAG Chatbot using Ollama Mistral")

# -------------------------------
# Session State Initialization
# -------------------------------
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False
if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = None

# -------------------------------
# Phase 1: Upload Document
# -------------------------------
if not st.session_state.document_processed:
    st.subheader("Upload a document to begin")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, TXT, DOCX, CSV)",
        type=["pdf", "txt", "docx", "csv"]
    )

    if uploaded_file:
        with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
            # Extract text
            text = extract_text(uploaded_file)
            
            if not text.strip():
                st.error("No text could be extracted from the file. Try another document.")
            else:
                # Chunk and embed
                chunks = chunk_text(text)
                embeddings = embed_chunks(chunks)
                index = build_faiss_index(embeddings)

                # Save to session state
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.document_processed = True
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.total_chunks = len(chunks)
                st.session_state.document_text = text  # Save full text for summarization

                # Success!
                st.success(f"Document '{uploaded_file.name}' indexed successfully!")
                st.info(f"**Total chunks created: {len(chunks)}**")
                st.rerun()

else:
    # -------------------------------
    # Phase 2: Chat Interface (Document Ready)
    # -------------------------------
    
    # Sidebar with document info and reset button
    with st.sidebar:
        st.success(f"📄 **{st.session_state.uploaded_file_name}**")
        st.info(f"Total chunks: {st.session_state.total_chunks}")
        
        st.markdown("---")
        
        # Document Summary button
        if st.button("📝 Document Summary", use_container_width=True):
            st.session_state.show_summary = True
            # Generate summary only once when button is clicked
            if st.session_state.generated_summary is None:
                with st.spinner("Generating summary..."):
                    st.session_state.generated_summary = document_summary(st.session_state.document_text)
            st.rerun()
        
        # Download chat history button
        if st.session_state.chat_history:
            chat_content = download_chat_history(st.session_state.chat_history)
            st.download_button(
                label="💾 Download Chat History",
                data=chat_content,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        if st.button("📤 Upload New Document", use_container_width=True):
            # Clear all session state
            st.session_state.document_processed = False
            st.session_state.chunks = None
            st.session_state.index = None
            st.session_state.chat_history = []
            st.session_state.document_text = None
            st.session_state.show_summary = False
            st.session_state.generated_summary = None
            if "uploaded_file_name" in st.session_state:
                del st.session_state.uploaded_file_name
            if "total_chunks" in st.session_state:
                del st.session_state.total_chunks
            st.rerun()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # -------------------------------
    # Display Document Summary (if requested)
    # -------------------------------
    if st.session_state.show_summary and st.session_state.generated_summary:
        with st.expander("📄 Document Summary", expanded=True):
            st.markdown(st.session_state.generated_summary)
            if st.button("Close Summary"):
                st.session_state.show_summary = False
                st.rerun()
    
    # -------------------------------
    # Display Chat History
    # -------------------------------
    st.subheader("💬 Conversation")
    
    if st.session_state.chat_history:
        for i, item in enumerate(st.session_state.chat_history):
            # User message
            with st.chat_message("user"):
                st.markdown(item['query'])
            
            # Assistant message
            with st.chat_message("assistant"):
                st.markdown(item['answer'])
    else:
        st.info("👋 Ask a question about your document to start the conversation!")

    # -------------------------------
    # Chat Input (Always visible at bottom)
    # -------------------------------
    st.markdown("---")
    
    # Create a form to prevent auto-rerun issues
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Your question:", label_visibility="collapsed", 
                                      placeholder="Type your question here...")
        with col2:
            submit_button = st.form_submit_button("Ask", use_container_width=True)

    if submit_button and user_input:
        with st.spinner("🤔 Thinking..."):
            # Retrieve relevant chunks
            top_chunks, top_scores = retrieve_and_rerank(
                user_input,
                st.session_state.index,
                st.session_state.chunks
            )

            # Build memory context from chat history
            memory_context = "\n".join([
                f"Q: {item['query']}\nA: {item['answer']}"
                for item in st.session_state.chat_history
            ])
            context_chunks = top_chunks + ([memory_context] if memory_context else [])

            # Generate answer
            answer = generate_answer_ollama(user_input, context_chunks)

        # Save to history
        st.session_state.chat_history.append({"query": user_input, "answer": answer})
        
        # Rerun to display the new message
        st.rerun()