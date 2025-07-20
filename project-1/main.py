# main.py

import os
os.environ["STREAMLIT_WATCHDOG"] = "False"

import streamlit as st
import whisper
from app.config import COCO_ANNOTATIONS_PATH
from app.image_loader import load_coco_annotations
from app.score_board import show_leaderboard
from app.game_logic import play_game
from PIL import Image
from app.yolo_detector import detect_image_objects

if "show_yolo" not in st.session_state:
    st.session_state.show_yolo = False

@st.cache_resource
def load_model():
    model = whisper.load_model("small")
    return model.to("cpu")

model = load_model()

# Load the Whisper model
image_objects = load_coco_annotations()

# Set up the sidebar UI
st.sidebar.title("Game Settings")
st.session_state.username = st.sidebar.text_input("Username", value="Player1")
st.session_state.rounds = st.sidebar.slider("Rounds", 1, 10, 3)
st.session_state.time_limit = st.sidebar.slider("Time limit (seconds)", 5, 15, 8)

# Set up the sidebar UI
if st.sidebar.button("Start Game"):
    st.session_state.show_yolo = False
    play_game(image_objects, rounds=st.session_state.rounds, is_mistake_mode=False, model=model)

if st.sidebar.button("Re-practice"):
    st.session_state.show_yolo = False
    play_game(image_objects, rounds=st.session_state.rounds, is_mistake_mode=True, model=model)

if st.sidebar.button("View ranking list"):
    show_leaderboard()

if st.sidebar.button("Reset status"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if st.sidebar.button("Exit the game"):
    st.write("👋Good-bye")

if st.sidebar.button("Image object recognition"):
    st.session_state.show_yolo = True

# yolo
if st.session_state.get("show_yolo", False):
    st.title("Image object recognition")
    uploaded_file = st.file_uploader("Please upload a picture for object recognition", type=["jpg", "jpeg", "png"], key="yolo_upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Pictures transmitted", use_container_width=True)
        with st.spinner("The object in the picture is being recognised, please wait..."):
            detected_classes = detect_image_objects(image)
        if detected_classes:
            st.success("The category of the detected object:")
            st.write(", ".join(detected_classes))
        else:
            st.warning("No objects detected.")