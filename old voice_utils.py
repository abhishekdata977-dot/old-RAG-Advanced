"""
Voice utilities for RAG Prototype - Enhanced and Fixed Version

This module provides speech recognition and text-to-speech functionality
with better Streamlit integration and error handling.

Author: RAG Prototype Team
Date: 2024
"""

import streamlit as st
import speech_recognition as sr
import tempfile
import os
import logging
import threading
import time
from typing import Optional
import uuid
import base64

# Configure logging
logger = logging.getLogger(__name__)

# Text-to-Speech using gTTS with better error handling
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    logger.info("gTTS available - TTS functionality enabled")
except ImportError as e:
    TTS_AVAILABLE = False
    logger.warning(f"gTTS not available: {e}")

# Audio playback support
try:
    import pygame
    PYGAME_AVAILABLE = True
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    logger.info("Pygame available - enhanced audio playback enabled")
except ImportError:
    PYGAME_AVAILABLE = False
    logger.info("Pygame not available - using Streamlit audio player only")

AUDIO_PLAYBACK_AVAILABLE = True  # Streamlit always supports audio playback

def initialize_voice_state():
    """Initialize voice-related session state variables"""
    if 'voice_text' not in st.session_state:
        st.session_state.voice_text = ""
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'voice_input_ready' not in st.session_state:
        st.session_state.voice_input_ready = False
    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = TTS_AVAILABLE
    if 'audio_language' not in st.session_state:
        st.session_state.audio_language = 'en'
    if 'audio_speed' not in st.session_state:
        st.session_state.audio_speed = False

def record_audio_from_mic(duration=5, sample_rate=16000):
    """
    Record audio from microphone using speech_recognition library
    
    Args:
        duration (int): Maximum recording duration in seconds
        sample_rate (int): Audio sample rate
        
    Returns:
        sr.AudioData: Recorded audio data or None if failed
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone(sample_rate=sample_rate) as source:
            # Create a placeholder for dynamic messages
            status_placeholder = st.empty()
            
            status_placeholder.info("🎤 Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 300
            
            status_placeholder.success("🎤 Listening... Start speaking now!")
            
            # Listen for audio with timeout
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=duration)
            status_placeholder.success("✅ Audio recorded successfully!")
            
            return audio
            
    except sr.WaitTimeoutError:
        st.error("⏰ No speech detected within timeout period")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Microphone error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Recording failed: {e}")
        return None

def transcribe_audio(audio_data):
    """
    Transcribe audio data to text using Google Speech Recognition
    
    Args:
        audio_data (sr.AudioData): Recorded audio data
        
    Returns:
        str: Transcribed text or None if failed
    """
    if not audio_data:
        return None
        
    recognizer = sr.Recognizer()
    
    try:
        # Use Google Web Speech API
        text = recognizer.recognize_google(audio_data, language=st.session_state.get('voice_language', 'en-US'))
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio - please try speaking more clearly")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition service error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return None

def create_voice_input_interface():
    """Create an enhanced voice input interface with better UX"""
    
    # Voice Input Section Header
    st.markdown("### 🎤 Voice Input")
    
    # Create columns for controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if not st.session_state.is_recording:
            if st.button("🎤 Start Voice Recording", type="primary", use_container_width=True):
                st.session_state.is_recording = True
                st.rerun()
        else:
            st.button("⏸️ Recording in Progress...", disabled=True, use_container_width=True)
    
    with col2:
        duration = st.selectbox("Duration", [3, 5, 8, 10], index=1, help="Recording duration in seconds")
    
    with col3:
        if st.button("🗑️ Clear Voice", use_container_width=True):
            st.session_state.voice_text = ""
            st.session_state.voice_input_ready = False
            st.session_state.is_recording = False
            st.rerun()
    
    # Handle recording process
    if st.session_state.is_recording:
        with st.spinner("🎤 Processing voice input..."):
            audio_data = record_audio_from_mic(duration=duration)
            
        if audio_data:
            with st.spinner("📝 Converting speech to text..."):
                transcribed_text = transcribe_audio(audio_data)
                
            if transcribed_text:
                st.session_state.voice_text = transcribed_text
                st.session_state.voice_input_ready = True
                st.success(f"✅ Voice recognized: **{transcribed_text}**")
            else:
                st.error("❌ Failed to transcribe audio. Please try again.")
        
        st.session_state.is_recording = False
        st.rerun()
    
    # Display recognized text with editing capability
    if st.session_state.voice_text:
        st.markdown("**Recognized Text:**")
        edited_text = st.text_area(
            "Edit if needed:", 
            value=st.session_state.voice_text,
            height=80,
            help="You can edit the recognized text before using it"
        )
        
        col_use, col_edit = st.columns([1, 1])
        with col_use:
            if st.button("✅ Use This Text", type="primary", use_container_width=True):
                return edited_text
        with col_edit:
            if edited_text != st.session_state.voice_text:
                if st.button("💾 Save Edits", use_container_width=True):
                    st.session_state.voice_text = edited_text
                    st.success("✅ Text updated!")
                    st.rerun()
    
    return None

def synthesize_speech(text: str, language: str = 'en', slow: bool = False) -> Optional[str]:
    """
    Convert text to speech using gTTS with improved error handling
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code (default: 'en')
        slow (bool): Whether to use slow speech rate
        
    Returns:
        str: Path to generated audio file or None if failed
    """
    if not TTS_AVAILABLE:
        st.warning("🔇 Text-to-speech not available. Install with: `pip install gtts`")
        return None
    
    if not text or not text.strip():
        return None
    
    # Clean and limit text length
    text = text.strip()
    max_length = 2000  # Increased limit
    if len(text) > max_length:
        text = text[:max_length] + "... (truncated for audio)"
        st.info(f"📝 Text truncated to {max_length} characters for audio generation")
    
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=slow)
        
        # Ensure temp directory exists
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename
        audio_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        # Save audio file
        tts.save(audio_path)
        
        if os.path.exists(audio_path):
            logger.info(f"TTS audio generated: {audio_path}")
            return audio_path
        else:
            st.error("❌ Failed to generate audio file")
            return None
            
    except Exception as e:
        st.error(f"❌ Text-to-speech generation failed: {e}")
        logger.error(f"TTS error: {e}")
        return None

def create_audio_player(audio_path: str, autoplay: bool = False):
    """
    Create an enhanced audio player widget for Streamlit
    
    Args:
        audio_path (str): Path to audio file
        autoplay (bool): Whether to autoplay the audio
        
    Returns:
        bool: True if audio player was created successfully
    """
    if not audio_path or not os.path.exists(audio_path):
        st.error("❌ Audio file not found")
        return False
    
    try:
        # Read audio file
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        # Create audio player with Streamlit
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
        
        # Add download option
        if st.button("💾 Download Audio", help="Download the generated audio file"):
            st.download_button(
                label="📥 Download MP3",
                data=audio_bytes,
                file_name=f"tts_audio_{int(time.time())}.mp3",
                mime="audio/mp3"
            )
        
        return True
        
    except Exception as e:
        st.error(f"❌ Audio player creation failed: {e}")
        logger.error(f"Audio player error: {e}")
        return False

def play_audio_with_pygame(audio_path: str):
    """
    Play audio using pygame in background (non-blocking)
    
    Args:
        audio_path (str): Path to audio file
    """
    if not PYGAME_AVAILABLE or not os.path.exists(audio_path):
        return False
    
    def play_in_background():
        try:
            pygame.mixer.music.stop()  # Stop any currently playing audio
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Pygame playback error: {e}")
    
    # Run in separate thread
    playback_thread = threading.Thread(target=play_in_background, daemon=True)
    playback_thread.start()
    return True

def create_tts_interface(text: str, auto_play: bool = False, show_controls: bool = True):
    """
    Create comprehensive text-to-speech interface
    
    Args:
        text (str): Text to convert to speech
        auto_play (bool): Automatically generate and play audio
        show_controls (bool): Show TTS controls
        
    Returns:
        str: Path to generated audio file or None
    """
    if not text or not text.strip():
        return None
    
    if not TTS_AVAILABLE:
        if show_controls:
            st.warning("🔇 Text-to-speech not available")
            st.info("Install with: `pip install gtts pygame`")
        return None
    
    audio_path = None
    
    if show_controls:
        st.markdown("### 🔊 Text-to-Speech")
        
        # TTS Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🎵 Generate Audio", type="primary", use_container_width=True):
                with st.spinner("🎵 Generating audio..."):
                    audio_path = synthesize_speech(
                        text, 
                        language=st.session_state.get('audio_language', 'en'),
                        slow=st.session_state.get('audio_speed', False)
                    )
                    
                if audio_path:
                    st.success("✅ Audio generated successfully!")
                    create_audio_player(audio_path)
                    
                    # Schedule cleanup
                    schedule_audio_cleanup(audio_path, delay=300)  # 5 minutes
        
        with col2:
            # Language selection
            languages = {
                'en': 'English',
                'es': 'Spanish', 
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh': 'Chinese'
            }
            
            selected_lang = st.selectbox(
                "Language",
                options=list(languages.keys()),
                format_func=lambda x: languages[x],
                index=0,
                key="tts_language_selector"
            )
            st.session_state.audio_language = selected_lang
        
        with col3:
            # Speed control
            st.session_state.audio_speed = st.checkbox(
                "Slow Speech",
                value=st.session_state.get('audio_speed', False),
                help="Enable slower speech rate"
            )
    
    # Auto-play functionality
    if auto_play:
        with st.spinner("🎵 Generating audio..."):
            audio_path = synthesize_speech(
                text,
                language=st.session_state.get('audio_language', 'en'),
                slow=st.session_state.get('audio_speed', False)
            )
        
        if audio_path:
            create_audio_player(audio_path, autoplay=True)
            schedule_audio_cleanup(audio_path, delay=300)
    
    return audio_path

def schedule_audio_cleanup(file_path: str, delay: int = 300):
    """
    Schedule cleanup of audio file after specified delay
    
    Args:
        file_path (str): Path to audio file
        delay (int): Delay in seconds before cleanup
    """
    def cleanup():
        time.sleep(delay)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up audio file: {file_path}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    threading.Thread(target=cleanup, daemon=True).start()

def create_voice_settings_interface():
    """Create comprehensive voice settings interface"""
    st.markdown("### ⚙️ Audio Configuration")
    
    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["🎤 Voice Input", "🔊 Speech Output", "📊 System Status"])
    
    with tab1:
        st.markdown("#### Speech Recognition Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Voice recognition language
            voice_languages = {
                'en-US': 'English (US)',
                'en-GB': 'English (UK)',
                'es-ES': 'Spanish (Spain)',
                'fr-FR': 'French (France)',
                'de-DE': 'German (Germany)',
                'it-IT': 'Italian (Italy)',
                'pt-BR': 'Portuguese (Brazil)',
                'ru-RU': 'Russian (Russia)',
                'ja-JP': 'Japanese (Japan)',
                'ko-KR': 'Korean (Korea)',
                'zh-CN': 'Chinese (Simplified)'
            }
            
            selected_voice_lang = st.selectbox(
                "Recognition Language",
                options=list(voice_languages.keys()),
                format_func=lambda x: voice_languages[x],
                index=0
            )
            st.session_state.voice_language = selected_voice_lang
        
        with col2:
            # Audio quality settings
            st.selectbox(
                "Audio Quality",
                ["Standard (16kHz)", "High (44.1kHz)"],
                index=0,
                help="Higher quality uses more processing power"
            )
            
            st.slider(
                "Noise Sensitivity", 
                min_value=100, 
                max_value=1000, 
                value=300, 
                step=50,
                help="Lower values = more sensitive to quiet speech"
            )
    
    with tab2:
        st.markdown("#### Text-to-Speech Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if TTS_AVAILABLE:
                st.success("✅ gTTS Available")
                
                # Test TTS functionality
                test_text = st.text_input("Test Text", value="Hello, this is a test of text-to-speech.")
                if st.button("🎵 Test TTS"):
                    if test_text:
                        audio_path = synthesize_speech(test_text)
                        if audio_path:
                            create_audio_player(audio_path)
            else:
                st.error("❌ gTTS Not Available")
                st.code("pip install gtts", language="bash")
        
        with col2:
            # Audio playback options
            st.markdown("**Playback Options:**")
            if PYGAME_AVAILABLE:
                st.success("✅ Pygame Available (Background Play)")
            else:
                st.info("ℹ️ Pygame not available - using Streamlit player only")
                st.code("pip install pygame", language="bash")
            
            st.success("✅ Streamlit Audio Player Available")
            
            # Auto-play setting
            st.session_state.tts_enabled = st.checkbox(
                "Enable Auto-play for Responses",
                value=st.session_state.get('tts_enabled', TTS_AVAILABLE),
                help="Automatically generate audio for AI responses"
            )
    
    with tab3:
        st.markdown("#### System Status")
        
        # System capabilities
        status_data = {
            "Speech Recognition": "✅ Available (speech_recognition + Google API)",
            "Text-to-Speech": "✅ Available (gTTS)" if TTS_AVAILABLE else "❌ Not Available",
            "Audio Playback": "✅ Available (Streamlit + Pygame)" if PYGAME_AVAILABLE else "✅ Available (Streamlit only)",
            "Background Audio": "✅ Supported" if PYGAME_AVAILABLE else "❌ Not Available",
        }
        
        for feature, status in status_data.items():
            st.markdown(f"**{feature}:** {status}")
        
        # Installation instructions
        st.markdown("#### 📦 Installation Commands")
        st.code("""
# Install all audio dependencies
pip install gtts pygame speechrecognition

# For microphone support (may require system audio drivers)
pip install pyaudio

# Alternative audio backend
pip install pydub simpleaudio
        """, language="bash")
        
        # Cleanup controls
        st.markdown("#### 🧹 Maintenance")
        if st.button("🗑️ Clean Temporary Audio Files"):
            temp_dir = os.path.join(os.getcwd(), "temp_audio")
            if os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    os.makedirs(temp_dir, exist_ok=True)
                    st.success("✅ Temporary audio files cleaned")
                except Exception as e:
                    st.error(f"❌ Cleanup failed: {e}")
            else:
                st.info("ℹ️ No temporary files to clean")

def get_manual_text_input():
    """Placeholder function for manual text input"""
    return None

# Utility function to get system audio info
def get_system_audio_info():
    """Get comprehensive audio system information"""
    return {
        "tts_available": TTS_AVAILABLE,
        "pygame_available": PYGAME_AVAILABLE,
        "audio_playback_available": AUDIO_PLAYBACK_AVAILABLE,
        "speech_recognition_available": True,  # speech_recognition is always imported
        "temp_audio_dir": os.path.join(os.getcwd(), "temp_audio")
    }

# Export the main functions for use in app.py
__all__ = [
    'initialize_voice_state',
    'create_voice_input_interface', 
    'synthesize_speech',
    'create_tts_interface',
    'create_audio_player',
    'create_voice_settings_interface',
    'get_system_audio_info',
    'TTS_AVAILABLE',
    'AUDIO_PLAYBACK_AVAILABLE'
]