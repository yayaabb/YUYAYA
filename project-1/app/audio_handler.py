# app/audio_handler.py

import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import time
import whisper
from app.config import AUDIO_FILENAME, SAMPLE_RATE, CHANNELS, WHISPER_MODEL_NAME

# load Whisper 
@st.cache_resource
def load_model():
    model = whisper.load_model(WHISPER_MODEL_NAME)
    return model.to("cpu")

# recording
def record_audio(duration=8, samplerate=SAMPLE_RATE, filename=AUDIO_FILENAME):
    st.write("🎤 The recording begins! Please tell me the items on the picture。")
    countdown_placeholder = st.sidebar.empty()
    for i in range(duration, 0, -1):
        countdown_placeholder.markdown(f"⏳ count down: {i} second(s)")
        time.sleep(1)
    countdown_placeholder.empty()

    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=CHANNELS, dtype=np.int16)
    sd.wait()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    st.write("✅ The recording ends and the recognition begins!")
    return filename

# recognize
def transcribe_audio(model, filename=AUDIO_FILENAME):
    result = model.transcribe(filename, condition_on_previous_text=False, fp16=False)
    recognized_text = result["text"].lower().strip()
    st.write("📝 Identify results:", recognized_text)
    return recognized_text.split()