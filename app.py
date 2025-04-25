# app.py
import streamlit as st
import joblib
from chat_parser import parse_chat, extract_features
import pandas as pd
from collections import Counter
import emoji
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

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

uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=["txt"])

if uploaded_file:
    df = parse_chat(uploaded_file)
    users = df['sender'].unique()
    selected_user = st.selectbox("Select a user to analyze", users)

    if selected_user:
        features = extract_features(df, selected_user)
        feature_df = pd.DataFrame([features])

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