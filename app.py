#20250819
"""
RAG Prototype - Streamlit Application (Enhanced Version)

This module provides a user-friendly web interface for the RAG pipeline
with improved voice input/output functionality and better layout.
"""

import streamlit as st
import os
import re
import uuid
import logging
import tempfile
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utilities
try:
    from utils import (
        InMemoryVectorStore, load_document, chunk_document, generate_embeddings,
        save_embeddings, load_embeddings, get_available_embedding_sets,
        generate_structured_answer_with_gemini, embedding_model,
        calculate_retrieval_metrics, calculate_response_metrics,
        calculate_system_performance_metrics, format_metrics_for_display,
        calculate_overall_system_accuracy, iterative_summarize_chunks,
        load_summary_cache, save_summary_cache, load_answer_cache,
        save_answer_cache, keyword_search, hybrid_search, rerank_chunks,
        rewrite_query, compress_context, estimate_tokens, refine_answer_with_gemini
    )
    logger.info("Successfully imported utilities")
except ImportError as e:
    logger.error(f"Failed to import utilities: {e}")
    st.error(f"Required utilities missing: {e}")
    st.stop()

# Import voice utilities
try:
    from voice_utils import (
        initialize_voice_state, create_voice_input_interface,
        create_tts_interface, create_voice_settings_interface,
        create_audio_player, get_system_audio_info,
        TTS_AVAILABLE, AUDIO_PLAYBACK_AVAILABLE, synthesize_speech,
        record_audio_from_mic, transcribe_audio
    )
    logger.info("Successfully imported voice utilities")
    VOICE_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Voice utilities not available: {e}")
    TTS_AVAILABLE = False
    AUDIO_PLAYBACK_AVAILABLE = False
    VOICE_FEATURES_AVAILABLE = False

# Google AI Configuration
import google.generativeai as genai

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    st.error("Google Gemini API Key not found in environment variables.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google Generative AI configured successfully")
except Exception as e:
    logger.error(f"Error configuring Google Generative AI: {e}")
    st.error(f"Error configuring Google Generative AI: {e}")
    st.stop()

st.set_page_config(
    page_title="RAG Prototype - Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .audio-section {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .questions-asked-box {
        text-align: center; 
        background: color(0,0,0,0)
        color: white;
    }
    /* Voice input button alignment */
    .voice-input-container {
        display: flex;
        align-items: flex-end;
        height: 100%;
        padding-top: 1.75rem;
    }
    .voice-input-container button {
        height: 2.5rem !important;
        width: 100% !important;
        margin-bottom: 0.2rem !important;
    }
    /* Ensure equal button sizes */
    div[data-testid="column"] button[key="load_btn"],
    div[data-testid="column"] button[key="delete_btn"] {
        height: 2.5rem !important;
        font-size: 0.9rem !important;
    }
    /* Align voice button with text input */
    div[data-testid="column"]:has(button:contains("Voice Input")) {
        display: flex;
        align-items: flex-end;
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>📚 RAG Prototype - Intelligent Document Q&A System</h1>
    <p>Enhanced with Advanced Voice Input/Output Capabilities</p>
</div>
""", unsafe_allow_html=True)

# Feature highlights - now as a dropdown
with st.expander("🚀 Key Features", expanded=False):
    st.markdown("""
    <div class="feature-box">
        <ul>
            <li>📄 <strong>Multi-format Support:</strong> Upload and process PDF/TXT documents</li>
            <li>🧠 <strong>AI-Powered Search:</strong> Semantic embeddings for intelligent document retrieval</li>
            <li>💬 <strong>Natural Conversation:</strong> Chat with your documents using natural language</li>
            <li>🎤 <strong>Voice Input:</strong> Hands-free queries with speech recognition</li>
            <li>🔊 <strong>Audio Responses:</strong> Text-to-speech for accessibility</li>
            <li>📊 <strong>Source Citations:</strong> View exact document references for every answer</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Session State Initialization
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore()
if 'embedding_model_loaded' not in st.session_state:
    st.session_state.embedding_model_loaded = embedding_model is not None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_embeddings_name' not in st.session_state:
    st.session_state.current_embeddings_name = None
if 'last_query_metrics' not in st.session_state:
    st.session_state.last_query_metrics = None
if 'tts_enabled' not in st.session_state:
    st.session_state.tts_enabled = TTS_AVAILABLE
if 'voice_input_text' not in st.session_state:
    st.session_state.voice_input_text = ""
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# Initialize voice state if available
if VOICE_FEATURES_AVAILABLE:
    initialize_voice_state()

# Sidebar - Document Processing
with st.sidebar:
    st.markdown("## 📄 Document Management")
    
    # System Status with better styling
    st.markdown("### 🔧 System Status")
    
    system_status = []
    if embedding_model is not None:
        system_status.append("🧠 AI Models: <span class='status-success'>Loaded</span>")
    else:
        system_status.append("🧠 AI Models: <span class='status-error'>Failed</span>")
    
    for status in system_status:
        st.markdown(status, unsafe_allow_html=True)
    
    st.divider()
    
    # Document Upload Section
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple documents to create a searchable knowledge base"
    )
    
    new_embedding_set_name = st.text_input(
        "Knowledge Base Name",
        placeholder="e.g., 'Technical Manuals', 'Research Papers'...",
        help="Give your document collection a descriptive name"
    )
    
    if st.button("🚀 Process Documents", disabled=not st.session_state.embedding_model_loaded):
        if uploaded_files and new_embedding_set_name.strip():
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing documents..."):
                all_chunks = []
                total_files = len(uploaded_files)
                
                try:
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        progress_bar.progress((i + 0.5) / total_files)
                        
                        # Save temporary file
                        os.makedirs("data", exist_ok=True)
                        temp_file_path = os.path.join("data", uploaded_file.name)
                        
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Process document
                        doc_id = str(uuid.uuid4())
                        text, structured_pages, toc_data, error = load_document(temp_file_path)
                        
                        if error:
                            st.error(f"Error loading {uploaded_file.name}: {error}")
                            continue
                        
                        chunks = chunk_document(text, doc_id, uploaded_file.name, 
                                              structured_pages=structured_pages, toc_data=toc_data)
                        chunks_with_embeddings = generate_embeddings(chunks)
                        all_chunks.extend(chunks_with_embeddings)
                        
                        # Cleanup
                        os.remove(temp_file_path)
                        
                        progress_bar.progress((i + 1) / total_files)
                    
                    # Save embeddings
                    if all_chunks:
                        status_text.text("Saving knowledge base...")
                        success, error = save_embeddings(all_chunks, new_embedding_set_name)
                        if success:
                            st.session_state.vector_store = InMemoryVectorStore()
                            st.session_state.vector_store.add_vectors(all_chunks)
                            st.session_state.current_embeddings_name = new_embedding_set_name
                            st.session_state.chat_history = []
                            
                            # Success message with details
                            st.success(f"✅ Successfully processed {len(uploaded_files)} files!")
                            st.info(f"📊 Generated {len(all_chunks)} searchable chunks")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"Error saving knowledge base: {error}")
                    
                    progress_bar.empty()
                    status_text.empty()
                
                except Exception as e:
                    st.error(f"Processing error: {e}")
                    progress_bar.empty()
                    status_text.empty()
        else:
            st.error("Please upload files and enter a knowledge base name.")
    
    st.divider()
    
    # Load Existing Embeddings
    st.markdown("### 📂 Load Existing Knowledge Base")
    available_sets = get_available_embedding_sets()
    
    if available_sets:
        selected_set = st.selectbox(
            "Choose knowledge base:", 
            available_sets,
            help="Select a previously processed document collection"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Selected", use_container_width=True, key="load_btn"):
                with st.spinner("Loading knowledge base..."):
                    embeddings_data, error = load_embeddings(selected_set)
                    
                    if error:
                        st.error(f"Loading error: {error}")
                    elif not embeddings_data:
                        st.error(f"No valid data in '{selected_set}'")
                    else:
                        st.session_state.vector_store = InMemoryVectorStore()
                        st.session_state.vector_store.add_vectors(embeddings_data)
                        st.session_state.current_embeddings_name = selected_set
                        st.success(f"✅ Loaded: {selected_set}")
                        st.rerun()
        
        with col2:
            if st.button("Delete", use_container_width=True, key="delete_btn"):
                try:
                    embeddings_file = f"embeddings/{selected_set}.pkl"
                    if os.path.exists(embeddings_file):
                        os.remove(embeddings_file)
                        st.success(f"✅ Deleted {selected_set}")
                        st.rerun()
                    else:
                        st.error(f"File not found: {embeddings_file}")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
    else:
        st.info("No knowledge bases found. Upload documents to get started.")
    
    st.divider()
    
    # Current Status
    st.markdown("### 📊 Current Status")
    if st.session_state.current_embeddings_name:
        st.info(f"**Active:** {st.session_state.current_embeddings_name}")
        st.info(f"**Chunks:** {len(st.session_state.vector_store.chunks_data):,}")
        
        # Calculate total content size
        total_chars = sum(len(chunk.get('content', '')) for chunk in st.session_state.vector_store.chunks_data)
        st.info(f"**Content:** {total_chars:,} characters")
    else:
        st.warning("No knowledge base loaded")



# Query Input Section with Voice Input
st.markdown('<h4 style="font-size: 1.2rem;">Chat Interface</h4>', unsafe_allow_html=True)

# Create input row with text input and voice button
col1, col2 = st.columns([4, 1])

with col1:
    user_query = st.text_input(
        "Ask your question:",
        value=st.session_state.voice_input_text,
        placeholder="Type your question here or use voice input...",
        help="Enter your question about the documents",
        key="main_query_input"
    )

with col2:
    if VOICE_FEATURES_AVAILABLE:
        # Add container div for better alignment
        st.markdown('<div class="voice-input-container">', unsafe_allow_html=True)
        
        if not st.session_state.is_recording:
            if st.button("🎤 Voice Input", use_container_width=True, type="primary"):
                st.session_state.is_recording = True
                st.rerun()
        else:
            st.button("🔴 Recording...", disabled=True, use_container_width=True)
            
            # Automatic voice recording with 13 seconds duration
            with st.spinner("Listening... (13 seconds)"):
                audio_data = record_audio_from_mic(duration=13)
                
            if audio_data:
                with st.spinner("Converting speech to text..."):
                    transcribed_text = transcribe_audio(audio_data)
                    
                if transcribed_text:
                    st.session_state.voice_input_text = transcribed_text
                    st.success(f"Voice recognized: **{transcribed_text}**")
                    st.session_state.is_recording = False
                    st.rerun()
                else:
                    st.error("Failed to transcribe audio. Please try again.")
            
            st.session_state.is_recording = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="voice-input-container">', unsafe_allow_html=True)
        st.info("Voice input not available")
        st.markdown('</div>', unsafe_allow_html=True)

# Process User Query
if user_query:
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Thinking and searching documents..."):
            if not st.session_state.embedding_model_loaded:
                assistant_response = "⚠ Embedding model not loaded. Please check system configuration."
            elif len(st.session_state.vector_store.chunks_data) == 0:
                assistant_response = "📚 No documents loaded. Please upload documents first to start chatting!"
            else:
                try:
                    # Enhanced query processing pipeline
                    query_embedding = embedding_model.encode([user_query])[0]
                    rewritten_query = rewrite_query(user_query)
                    
                    # Hybrid search for better results
                    candidate_chunks = hybrid_search(
                        rewritten_query,
                        st.session_state.vector_store,
                        st.session_state.vector_store.chunks_data,
                        embedding_model,
                        k_semantic=7,
                        k_keyword=7
                    )
                    
                    # Rerank for relevance
                    retrieved_chunks = rerank_chunks(rewritten_query, candidate_chunks, top_k_reranked=7)
                    
                    if retrieved_chunks:
                        # Generate initial response
                        initial_response = generate_structured_answer_with_gemini(
                            rewritten_query, retrieved_chunks, query_embedding,
                            st.session_state.vector_store.chunks_data, rerank_top_k=7
                        )
                        
                        # Refine the response for better human-like quality
                        assistant_response = refine_answer_with_gemini(initial_response)
                        
                        # Calculate and store metrics
                        retrieval_metrics = calculate_retrieval_metrics(user_query, retrieved_chunks, query_embedding)
                        response_metrics = calculate_response_metrics(assistant_response, user_query, retrieved_chunks)
                        
                        st.session_state.last_query_metrics = {
                            'retrieval': retrieval_metrics,
                            'response': response_metrics
                        }
                    else:
                        assistant_response = "🔍 I couldn't find relevant information in the loaded documents. Please try rephrasing your question or check if the relevant documents are loaded."
                
                except Exception as e:
                    logger.error(f"Query processing error: {e}")
                    assistant_response = f"⚠ An error occurred while processing your query: {str(e)}"

            # Display the response
            st.markdown(assistant_response)
            
            # Add to chat history (user message first, then assistant response)
            st.session_state.chat_history.insert(0, {"role": "assistant", "content": assistant_response})
            st.session_state.chat_history.insert(0, {"role": "user", "content": user_query})
            
            # Clear voice input text after processing
            st.session_state.voice_input_text = ""

            # Generate audio response if enabled
            if st.session_state.tts_enabled and TTS_AVAILABLE and assistant_response:
                st.markdown("---")
                st.markdown("### Audio Response")
                
                with st.spinner("🎵 Generating audio response..."):
                    try:
                        audio_path = synthesize_speech(
                            assistant_response,
                            language=st.session_state.get('audio_language', 'en'),
                            slow=st.session_state.get('audio_speed', False)
                        )
                        
                        if audio_path:
                            create_audio_player(audio_path)
                            st.success("✅ Audio generated successfully!")
                        else:
                            st.warning("⚠️ Could not generate audio response")
                    
                    except Exception as e:
                        st.error(f"⚠ Audio generation failed: {e}")

# Display chat history (newest first)
for i, message in enumerate(st.session_state.chat_history):
    if message["content"]:
        with st.chat_message(message["role"], avatar="🙋‍♂️" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

# Performance Metrics Display
if st.session_state.last_query_metrics:
    with st.expander("📊 Query Performance Metrics", expanded=False):
        metrics = st.session_state.last_query_metrics
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Retrieval Quality")
            retrieval = metrics['retrieval']
            st.metric("Chunks Retrieved", retrieval['num_chunks_retrieved'])
            st.metric("Average Similarity", f"{retrieval['avg_similarity']:.3f}")
            st.metric("Retrieval Accuracy", f"{retrieval['retrieval_accuracy']:.1%}")
            st.metric("Semantic Coherence", f"{retrieval['semantic_coherence']:.3f}")
        
        with col2:
            st.markdown("#### 💬 Response Quality")
            response = metrics['response']
            st.metric("Response Length", f"{response['response_word_count']} words")
            st.metric("Context Utilization", f"{response['context_utilization']:.1%}")
            st.metric("Answer Completeness", f"{response['answer_completeness']:.1%}")
            st.metric("Response Accuracy", f"{response['response_accuracy']:.1%}")

# Chat Management
st.markdown('<h5 style="font-size: 1rem;">Chat Management</h5>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("✅ Chat history cleared!")
        st.rerun()

with col2:
    if st.button("Export Chat", use_container_width=True):
        if st.session_state.chat_history:
            chat_export = ""
            for msg in st.session_state.chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_export += f"**{role}:** {msg['content']}\n\n"
            
            timestamp = int(time.time())
            st.download_button(
                label="Download Chat",
                data=chat_export,
                file_name=f"chat_export_{timestamp}.txt",
                mime="text/plain"
            )
        else:
            st.info("No chat history to export")

with col3:
    chat_count = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
    st.markdown(f'<div class="questions-asked-box"><strong>Questions Asked</strong><br><span style="font-size: 1.5rem;">{chat_count}</span></div>', unsafe_allow_html=True)

st.divider()
# Text-to-Speech Settings (moved up)
if TTS_AVAILABLE:
    with st.expander("Text-to-Speech Settings", expanded=False):
        st.markdown("#### Audio Response Settings")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.session_state.tts_enabled = st.checkbox(
                "Enable Audio Responses", 
                value=st.session_state.tts_enabled,
                help="Automatically generate speech for AI responses"
            )
        
        with col2:
            if st.session_state.tts_enabled:
                audio_language = st.selectbox(
                    "",
                    options=['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'],
                    format_func=lambda x: {
                        'en': 'English', 'es': 'Spanish', 'fr': 'French', 
                        'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
                        'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese'
                    }[x],
                    index=0
                )
                st.session_state.audio_language = audio_language
        
        with col3:
            if st.session_state.tts_enabled:
                st.session_state.audio_speed = st.checkbox(
                    "Slow Speech",
                    value=st.session_state.get('audio_speed', False),
                    help="Enable slower speech rate for better comprehension"
                )

        # Advanced Audio Settings Panel (below TTS settings)
        
if VOICE_FEATURES_AVAILABLE:
            with st.expander("Advanced Audio Settings", expanded=False):
                create_voice_settings_interface()
# System Information
with st.expander("System Information & Troubleshooting", expanded=False):
    st.markdown("### System Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Features:**")
        features = [
            ("Speech Recognition", VOICE_FEATURES_AVAILABLE, "speech_recognition + Google API"),
            ("Text-to-Speech", TTS_AVAILABLE, "gTTS (Google Text-to-Speech)"),
            ("Audio Playback", AUDIO_PLAYBACK_AVAILABLE, "Streamlit native player"),
            ("AI Models", embedding_model is not None, "SentenceTransformer + CrossEncoder"),
            ("Document Processing", True, "PyPDF2 + Text processing")
        ]
        
        for feature, available, tech in features:
            status = "✅" if available else "⚠"
            st.write(f"{status} **{feature}:** {tech}")
    
    with col2:
        st.markdown("**Installation Commands:**")
        st.code("""
# Core RAG functionality
pip install streamlit sentence-transformers
pip install google-generativeai
pip install PyPDF2 scikit-learn numpy

# Voice features
pip install gtts speechrecognition pygame
pip install pyaudio  # For microphone input

# Optional enhancements
pip install rank-bm25 cross-encoder
        """, language="bash")
    
    st.markdown("### Troubleshooting")
    
    troubleshooting_tips = [
        "**No audio output?** Ensure gtts is installed: `pip install gtts`",
        "**Microphone not working?** Install pyaudio: `pip install pyaudio`",
        "**Slow processing?** Try smaller document chunks or fewer documents",
        "**API errors?** Check your GEMINI_API_KEY environment variable",
        "**Import errors?** Reinstall dependencies: `pip install -r requirements.txt`"
    ]
    
    for tip in troubleshooting_tips:
        st.markdown(f"• {tip}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <h4>🚀 RAG Prototype with Advanced Voice I/O</h4>
    <p><em>Empowering intelligent document interaction through voice and text</em></p>
</div>
""", unsafe_allow_html=True)