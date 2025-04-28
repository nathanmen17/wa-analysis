   # app.py
import streamlit as st
import joblib
from chat_parser import parse_chat, extract_features
import pandas as pd
from collections import Counter
import emoji
import nltk
from nltk.corpus import stopwords
import zipfile
import os

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stop_words.update(["media", "omitted"]) # Adding the new stop words

def analyze_usage(df, user):
    user_df = df[df['sender'] == user].copy()
    all_words = ' '.join(user_df['message']).lower().split()
    filtered_words = [word for word in all_words if word.isalnum() and word not in stop_words]
    most_common_words = Counter(filtered_words).most_common(5)
    emojis_list = [c for msg in user_df['message'] for c in msg if c in emoji.EMOJI_DATA]
    most_common_emojis = Counter(emojis_list).most_common(5)
    return most_common_words, most_common_emojis

st.title("WhatsApp Interest Predictor (ML Powered)")
st.write("Upload a chat, select a user, and see how interested they are in %!")

upload_option = st.selectbox(
    "Select upload type:",
    ("Single .txt file", ".zip file containing .txt files")
)

uploaded_file = None
chat_file = None
df = None

if upload_option == "Single .txt file":
    uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=["txt"])
    if uploaded_file:
        df = parse_chat(uploaded_file)
        chat_file = uploaded_file.name
elif upload_option == ".zip file containing .txt files":
    uploaded_file = st.file_uploader("Upload .zip file containing WhatsApp Chats", type=["zip"])
    if uploaded_file:
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                text_files = [f for f in zip_ref.namelist() if f.endswith(".txt")]
                if text_files:
                    selected_chat_file = st.selectbox("Select a chat file from the zip:", text_files)
                    if selected_chat_file:
                        with zip_ref.open(selected_chat_file) as f:
                            df = parse_chat(f)
                            chat_file = selected_chat_file
                else:
                    st.error("No .txt files found in the uploaded zip file.")
        except zipfile.BadZipFile:
            st.error("Invalid zip file.")

if df is not None:
    users = df['sender'].unique()
    selected_user = st.selectbox("Select a user to analyze", users)

    if selected_user:
        st.subheader(f"Analyzing chat: {chat_file}")
        features = extract_features(df, selected_user)
        feature_df = pd.DataFrame([features])

        try:
            model = joblib.load('interest_model.pkl')
            prediction = model.predict(feature_df)[0]

            st.subheader(f"Predicted Interest Score for {selected_user}: {round(prediction, 2)}%")
            st.markdown("### Feature Breakdown")
            for k, v in features.items():
                st.write(f"**{k.replace('_', ' ').title()}**: {round(v, 2)}")

            # Analyze and display word and emoji usage
            most_common_words, most_common_emojis = analyze_usage(df, selected_user)

            st.markdown("### Most Frequent Words (excluding common words)")
            if most_common_words:
                for word, count in most_common_words:
                    st.write(f"- {word}: {count}")
            else:
                st.write("No significant words found after filtering.")

            st.markdown("### Most Frequent Emojis")
            if most_common_emojis:
                for emo, count in most_common_emojis:
                    st.write(f"- {emo}: {count}")
            else:
                st.write("No emojis used by this user.")

        except FileNotFoundError:
            st.error("Error: 'interest_model.pkl' not found. Make sure the model file is in the same directory as the app.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")