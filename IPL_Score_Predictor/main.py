import streamlit as st
import numpy as np
import pickle

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ipl_score_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏏 IPL Score Predictor")

venue = st.selectbox("Select Venue", label_encoders['venue'].classes_)
batting_team = st.selectbox("Select Batting Team", label_encoders['bat_team'].classes_)
bowling_team = st.selectbox("Select Bowling Team", label_encoders['bowl_team'].classes_)
striker = st.selectbox("Select Striker", label_encoders['batsman'].classes_)
bowler = st.selectbox("Select Bowler", label_encoders['bowler'].classes_)

runs = st.number_input("Runs", min_value=0, step=1, value=0)
wickets = st.number_input("Wickets", min_value=0, step=1, value=0)
overs = st.number_input("Overs", min_value=0.0, step=0.1, value=0.0, format="%.1f")
striker_ind = st.selectbox("Striker Indicator (0 or 1)", [0, 1])

if st.button("Predict Score"):
    encoded_venue = label_encoders['venue'].transform([venue])[0]
    encoded_batting_team = label_encoders['bat_team'].transform([batting_team])[0]
    encoded_bowling_team = label_encoders['bowl_team'].transform([bowling_team])[0]
    encoded_striker = label_encoders['batsman'].transform([striker])[0]
    encoded_bowler = label_encoders['bowler'].transform([bowler])[0]

    input_features = [
        encoded_batting_team,
        encoded_striker,

        encoded_bowling_team,
        encoded_bowler,
        runs,
        wickets,
        encoded_venue,
        overs,
        striker_ind,
    ]

    input_array = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    predicted_score = model.predict(input_scaled)[0]
    st.success(f"🎯 Predicted Total Runs: {int(predicted_score)}")
