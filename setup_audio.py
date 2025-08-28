import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    packages = [
        "streamlit",
        "streamlit-webrtc", 
        "SpeechRecognition",
        "google-cloud-texttospeech",
        "pygame"
        "langchain-google-genai"
        "sounddevice"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def setup_directories():
    """Create necessary directories"""
    dirs = ["temp", "data", "embeddings"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Created directory: {d}")

def check_environment():
    """Check environment configuration"""
    print("\n🔍 Environment Check:")
    
    # Check Google Cloud credentials
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        print("✅ Google Cloud credentials configured")
    else:
        print("❌ Google Cloud credentials not found")
        print("   Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    
    # Check Gemini API key
    if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'):
        print("✅ Gemini API key configured")
    else:
        print("❌ Gemini API key not found")
        print("   Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

if __name__ == "__main__":
    print("🔧 Setting up RAG Prototype Audio Features...")
    
    install_requirements()
    setup_directories()
    check_environment()
    
    print("\n✅ Setup complete!")
    print("Run: streamlit run app.py")