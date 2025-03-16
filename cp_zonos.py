import streamlit as st
import torch
import whisper
import os
import tempfile
import numpy as np
from TTS.api import TTS
import soundfile as sf
import nest_asyncio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig  # Add this import
from gtts import gTTS
from io import BytesIO
from audiorecorder import audiorecorder
import time

# Apply nest_asyncio to avoid event loop issues
nest_asyncio.apply()

# Fix PyTorch 2.6 UnpicklingError Issue - Add all required classes
torch.serialization.add_safe_globals([
    XttsConfig, 
    XttsAudioConfig,
    BaseDatasetConfig
])

# Set page configuration
st.set_page_config(
    page_title="Speech Converter with Whisper & gTTS",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #9333ea, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #4b5563;
    }
    .stAudio {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
    }
    .model-loading {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #000000;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #dcfce7;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Create a models directory
MODELS_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize session state variables
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'tts_model' not in st.session_state:
    st.session_state.tts_model = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'voice_samples' not in st.session_state:
    st.session_state.voice_samples = {}

# App header
st.markdown('<h1 class="main-header">Speech Converter with Whisper & gTTS</h1>', unsafe_allow_html=True)
st.markdown("A bidirectional speech converter using OpenAI's Whisper for speech recognition and Google Text-to-Speech for synthesis")

# Sidebar for model selection
st.sidebar.header("Model Settings")

whisper_model_size = st.sidebar.selectbox(
    "Select Whisper Model Size",
    ["tiny", "base", "small", "medium"],
    index=1,
    help="Larger models are more accurate but slower and require more memory"
)

# Load Whisper model
@st.cache_resource
def load_whisper_model(model_size):
    try:
        with st.spinner(f"Loading Whisper {model_size} model... This might take a while."):
            return whisper.load_model(model_size)
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

# Load TTS model - We'll skip this for now and use gTTS instead
def load_tts_model():
    st.warning("XTTS model loading is currently disabled due to PyTorch 2.6 compatibility issues. Using Google Text-to-Speech instead.")
    return None

# Load models button
if st.sidebar.button("Load/Reload Models"):
    with st.sidebar:
        st.session_state.whisper_model = load_whisper_model(whisper_model_size)
        if st.session_state.whisper_model:
            st.success(f"Whisper {whisper_model_size} model loaded successfully!")
        
        # We'll skip loading the XTTS model for now
        st.session_state.tts_model = None
        st.info("Using Google Text-to-Speech (gTTS) for speech synthesis")

# TTS settings
st.sidebar.header("Text-to-Speech Settings")

languages = {
    "English (US)": "en",
    "English (UK)": "en-gb",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-CN",
    "Hindi": "hi"
}

selected_language = st.sidebar.selectbox("TTS Language", list(languages.keys()))
language_code = languages[selected_language]

# Speech speed
speech_speed = st.sidebar.slider("Speech Speed", min_value=False, max_value=True, value=False, 
                               format="Slow" if False else "Normal")

# Main app layout with tabs
tab1, tab2 = st.tabs(["Speech to Text (Whisper AI)", "Text to Speech (gTTS)"])

# Speech to Text tab
with tab1:
    st.markdown('<h2 class="subheader">Convert Speech to Text with Whisper AI</h2>', unsafe_allow_html=True)
    
    if st.session_state.whisper_model is None:
        st.markdown("""
        <div class="model-loading">
            <strong>⚠️ Whisper model not loaded!</strong><br>
            Please load the Whisper model from the sidebar before proceeding.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>Instructions:</strong>
            <ol>
                <li>Upload an audio file or record audio directly</li>
                <li>Click "Transcribe" to convert speech to text</li>
                <li>The transcription will appear below and can be sent to the Text-to-Speech tab</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Option to upload audio file or record
        option = st.radio("Choose input method:", ["Upload Audio File", "Record Audio"])
        
        audio_file = None
        
        if option == "Upload Audio File":
            audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
            
            if audio_file is not None:
                st.audio(audio_file, format="audio/wav")
        else:
            st.write("Click to record audio:")
            audio_bytes = audiorecorder("Click to record", "Recording...")
            
            if len(audio_bytes) > 0:
                st.audio(audio_bytes, format="audio/wav")
                audio_file = audio_bytes
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Transcribe Audio", use_container_width=True):
                if audio_file is None:
                    st.error("Please upload or record an audio file first!")
                else:
                    with st.spinner("Transcribing audio... This might take a while."):
                        # Save uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            if option == "Upload Audio File":
                                tmp_file.write(audio_file.read())
                            else:
                                tmp_file.write(audio_file)
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Transcribe the audio file
                            result = st.session_state.whisper_model.transcribe(tmp_file_path)
                            transcription = result["text"]
                            
                            # Store the transcription in session state
                            st.session_state.transcription = transcription
                            
                            # Store the audio path for voice cloning
                            st.session_state.audio_path = tmp_file_path
                            
                            st.success("Transcription complete!")
                        except Exception as e:
                            st.error(f"Error during transcription: {e}")
                            # Clean up the temporary file
                            os.unlink(tmp_file_path)
        
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.transcription = ""
                if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                    try:
                        os.unlink(st.session_state.audio_path)
                    except:
                        pass
                st.session_state.audio_path = None
                st.experimental_rerun()
        
        # Display transcription
        st.markdown("### Transcription Result:")
        transcription = st.text_area("", st.session_state.transcription, height=200)
        
        # Button to send transcription to TTS tab
        if st.button("Send to Text-to-Speech"):
            st.session_state.transcription = transcription
            st.experimental_rerun()

# Text to Speech tab
with tab2:
    st.markdown('<h2 class="subheader">Convert Text to Speech with Google TTS</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Instructions:</strong>
        <ol>
            <li>Enter text or use transcription from Whisper AI</li>
            <li>Select language and speech speed from the sidebar</li>
            <li>Click "Generate Speech" to convert text to audio</li>
            <li>Download the generated audio file</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Text input
    text_input = st.text_area("Enter text to convert to speech:", 
                            st.session_state.transcription, 
                            height=200)
    
    col1, col2 = st.columns(2)
    
    # Generate speech button
    if col1.button("Generate Speech", use_container_width=True):
        if text_input:
            try:
                with st.spinner("Generating speech..."):
                    # Create a BytesIO object to store the audio file
                    audio_bytes = BytesIO()
                    
                    # Generate the speech
                    tts = gTTS(text=text_input, lang=language_code.split('-')[0], slow=speech_speed)
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)
                    
                    # Play the audio
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Speech generated successfully!</strong><br>
                        You can download the audio file using the button below.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name="generated_speech.mp3",
                        mime="audio/mp3"
                    )
            except Exception as e:
                st.error(f"Error generating speech: {e}")
        else:
            st.warning("Please enter some text to convert to speech.")
    
    with col2:
        if st.button("Clear Text", use_container_width=True):
            st.session_state.transcription = ""
            st.experimental_rerun()

