# app/score_board.py

import json
import os
import streamlit as st
from app.config import SCORE_FILE

def load_scores():
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_score(username, score):
    data = load_scores()
    data[username] = max(data.get(username, 0), score)
    with open(SCORE_FILE, "w") as f:
        json.dump(data, f)

def show_leaderboard():
    data = load_scores()
    if not data:
        st.write("No historical results yet")
        return

    st.write("**🏆 Ranking List**")
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    for i, (user, sc) in enumerate(sorted_data, start=1):
        st.write(f"{i}. {user}: {sc} points")