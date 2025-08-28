"""
Simple Audio Utilities for RAG Prototype

Provides basic audio functionality with minimal dependencies.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Check for basic audio support
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

AUDIO_PLAYBACK_AVAILABLE = PYGAME_AVAILABLE

def initialize_audio_system():
    """Initialize audio system"""
    if PYGAME_AVAILABLE:
        try:
            pygame.mixer.init()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
    return False

def get_system_audio_info():
    """Get audio system information"""
    return {
        "pygame_available": PYGAME_AVAILABLE,
        "gtts_available": GTTS_AVAILABLE,
        "audio_playback_available": AUDIO_PLAYBACK_AVAILABLE
    }

# Keep only essential functions - remove redundant TTS functions
# The main TTS functionality is now in voice_utils.py