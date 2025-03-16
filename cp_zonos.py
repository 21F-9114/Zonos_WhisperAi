import streamlit as st
import torch
import whisper
from TTS.api import TTS  # Using Zonos TTS by Zyphra AI
import soundfile as sf
import numpy as np

# Load Whisper model
st.title("Speech-to-Text and Text-to-Speech App")
whisper_model = whisper.load_model("base")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Transcribe using Whisper
    result = whisper_model.transcribe(audio_path)
    text = result["text"]
    st.write("### Transcription:", text)
    
    # Text-to-Speech using Zonos by Zyphra AI
    tts_model = TTS("zonos/en/expressive")  # Using Zonos expressive model
    tts_audio_path = "generated_speech.wav"
    tts_model.tts_to_file(text=text, file_path=tts_audio_path)
    
    # Play generated speech
    st.audio(tts_audio_path, format="audio/wav")
