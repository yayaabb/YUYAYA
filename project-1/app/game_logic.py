# app/game_logic.py

import streamlit as st
import os
import time
import random
import matplotlib.image as mpimg
from app.audio_handler import record_audio, transcribe_audio
from app.utils_text import fuzzy_process_words, normalize
from app.score_board import save_score
from app.config import COCO_IMAGES_DIR

def play_game(image_objects, rounds=3, is_mistake_mode=False, model=None):
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "mistakes" not in st.session_state:
        st.session_state.mistakes = []
    if "mistake_mode" not in st.session_state:
        st.session_state.mistake_mode = False

    if not is_mistake_mode:
        st.session_state.score = 0
        st.session_state.mistakes = []

    if is_mistake_mode:
        if not st.session_state.mistakes:
            st.warning("❗ No mistakes to practice.")
            return
        image_list = [m[0] for m in st.session_state.mistakes]
    else:
        image_list = list(image_objects.keys())

    if not image_list:
        st.error("No images found.")
        return

    st.title("🎮 Picture-to-word Vocabulary Challenge")
    image_placeholder = st.empty()
    score_placeholder = st.sidebar.empty()

    if is_mistake_mode:
        mistakes_placeholder = st.sidebar.empty()
        actual_rounds = min(rounds, len(image_list))
    else:
        actual_rounds = rounds

    for _ in range(actual_rounds):
        if is_mistake_mode:
            mistakes_placeholder.subheader(f"📝 Remaining: {len(image_list)}")

        if not image_list:
            break

        random_image = random.choice(image_list)
        correct_set = image_objects[random_image]

        img_path = os.path.join(COCO_IMAGES_DIR, random_image)
        img_data = mpimg.imread(img_path)
        image_placeholder.image(img_data, caption="📸 What items are in the picture?", use_container_width=True)

        if not is_mistake_mode:
            score_placeholder.subheader(f"🏆 Current Score: {st.session_state.score}")

        audio_filename = record_audio(duration=st.session_state.time_limit)
        recognized_words = transcribe_audio(model, audio_filename)

        matched = fuzzy_process_words(recognized_words, correct_set)

        if not is_mistake_mode:
            st.session_state.score += len(matched)
            score_placeholder.subheader(f"🏆 Current Score: {st.session_state.score}")

        if matched:
            st.success(f"✅ Matched: {', '.join(matched)} (+{len(matched)} points)")
        else:
            st.error("❌ No correct words recognized.")

        norm_correct = set(normalize(w) for w in correct_set)
        missed = norm_correct - matched
        if missed:
            st.session_state.mistakes.append((random_image, list(missed)))
            st.error(f"📕 Missed: {', '.join(missed)}")

        if is_mistake_mode:
            image_list.remove(random_image)

        time.sleep(2)

    if not is_mistake_mode:
        st.subheader(f"🎉 Game Over! Final Score: {st.session_state.score}")
        if st.session_state.mistakes:
            st.subheader("📕 Mistakes Summary")
            for img, words in st.session_state.mistakes:
                st.write(f"📸 {img}: {', '.join(words)}")
        save_score(st.session_state.username, st.session_state.score)
    else:
        st.subheader("✅ Mistake Practice Finished")
        if st.session_state.mistakes:
            st.subheader("📕 Still Missed:")
            for img, words in st.session_state.mistakes:
                st.write(f"📸 {img}: {', '.join(words)}")