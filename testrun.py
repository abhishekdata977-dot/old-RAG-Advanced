import os
import streamlit as st
from voice_utils import (
    initialize_voice_state, create_voice_input_interface,
    create_tts_interface, TTS_AVAILABLE
)

# Initialize voice state
initialize_voice_state()

st.title("RAG Prototype - Voice & Manual Input Test")

# Manual query input (always available)
st.subheader("⌨️ Manual Input")
manual_query = st.text_input("Type your question here...")

# Voice input (optional)
with st.expander("🎤 Voice Input", expanded=False):
    voice_text = create_voice_input_interface()

# Determine query source
user_query = manual_query or voice_text

if user_query:
    st.markdown(f"**You asked:** {user_query}")

    # Dummy response for testing
    assistant_response = f"This is the response to: {user_query}"
    st.markdown(assistant_response)

    # Audio output directly in Streamlit without saving to Documents
    if TTS_AVAILABLE:
        with st.spinner("🔊 Generating audio..."):
            audio_file = create_tts_interface(assistant_response, auto_play=False)
            if audio_file and os.path.exists(audio_file):
                with open(audio_file, 'rb') as af:
                    st.audio(af.read(), format='audio/mp3')