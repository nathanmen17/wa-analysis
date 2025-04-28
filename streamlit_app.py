#    # app.py
# import streamlit as st
# import joblib
# from chat_parser import parse_chat, extract_features
# import pandas as pd
# from collections import Counter
# import emoji
# import nltk
# from nltk.corpus import stopwords
# import zipfile
# import os

# nltk.download('stopwords', quiet=True)
# stop_words = set(stopwords.words('english'))
# stop_words.update(["media", "omitted"]) # Adding the new stop words

# def analyze_usage(df, user):
#     user_df = df[df['sender'] == user].copy()
#     all_words = ' '.join(user_df['message']).lower().split()
#     filtered_words = [word for word in all_words if word.isalnum() and word not in stop_words]
#     most_common_words = Counter(filtered_words).most_common(5)
#     emojis_list = [c for msg in user_df['message'] for c in msg if c in emoji.EMOJI_DATA]
#     most_common_emojis = Counter(emojis_list).most_common(5)
#     return most_common_words, most_common_emojis

# st.title("WhatsApp Interest Predictor (ML Powered)")
# st.write("Upload a chat, select a user, and see how interested they are in %!")

# upload_option = st.selectbox(
#     "Select upload type:",
#     ("Single .txt file", ".zip file containing .txt files")
# )

# uploaded_file = None
# chat_file = None
# df = None

# if upload_option == "Single .txt file":
#     uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=["txt"])
#     if uploaded_file:
#         df = parse_chat(uploaded_file)
#         chat_file = uploaded_file.name
# elif upload_option == ".zip file containing .txt files":
#     uploaded_file = st.file_uploader("Upload .zip file containing WhatsApp Chats", type=["zip"])
#     if uploaded_file:
#         try:
#             with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
#                 text_files = [f for f in zip_ref.namelist() if f.endswith(".txt")]
#                 if text_files:
#                     selected_chat_file = st.selectbox("Select a chat file from the zip:", text_files)
#                     if selected_chat_file:
#                         with zip_ref.open(selected_chat_file) as f:
#                             df = parse_chat(f)
#                             chat_file = selected_chat_file
#                 else:
#                     st.error("No .txt files found in the uploaded zip file.")
#         except zipfile.BadZipFile:
#             st.error("Invalid zip file.")

# if df is not None:
#     users = df['sender'].unique()
#     selected_user = st.selectbox("Select a user to analyze", users)

#     if selected_user:
#         st.subheader(f"Analyzing chat: {chat_file}")
#         features = extract_features(df, selected_user)
#         feature_df = pd.DataFrame([features])

#         try:
#             model = joblib.load('interest_model.pkl')
#             prediction = model.predict(feature_df)[0]

#             st.subheader(f"Predicted Interest Score for {selected_user}: {round(prediction, 2)}%")
#             st.markdown("### Feature Breakdown")
#             for k, v in features.items():
#                 st.write(f"**{k.replace('_', ' ').title()}**: {round(v, 2)}")

#             # Analyze and display word and emoji usage
#             most_common_words, most_common_emojis = analyze_usage(df, selected_user)

#             st.markdown("### Most Frequent Words (excluding common words)")
#             if most_common_words:
#                 for word, count in most_common_words:
#                     st.write(f"- {word}: {count}")
#             else:
#                 st.write("No significant words found after filtering.")

#             st.markdown("### Most Frequent Emojis")
#             if most_common_emojis:
#                 for emo, count in most_common_emojis:
#                     st.write(f"- {emo}: {count}")
#             else:
#                 st.write("No emojis used by this user.")

#         except FileNotFoundError:
#             st.error("Error: 'interest_model.pkl' not found. Make sure the model file is in the same directory as the app.")
#         except Exception as e:
#             st.error(f"An error occurred during prediction: {e}")
# app.py# app.py
# app.pyimport 
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

def get_emoji_for_percentage(percentage):
    if percentage < 20:
        return "😞"  # Sad
    elif percentage < 40:
        return "😐"  # Neutral
    elif percentage < 60:
        return "😊"  # Slightly happy
    elif percentage < 80:
        return "😄"  # Happy
    else:
        return "🎉"  # Fun/Excited

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
            predicted_percentage = round(prediction, 2)
            emoji_for_prediction = get_emoji_for_percentage(predicted_percentage)
            st.subheader(f"Predicted Interest Score for {selected_user}: {predicted_percentage}% {emoji_for_prediction}")


            st.markdown(f"""
                <style>
                    .progress-bar-container {{
                        background-color: #f0f2f6;
                        border-radius: 5px;
                        height: 25px;
                        width: 100%;
                        overflow: hidden;
                        position: relative; /* For absolute positioning of marks */
                    }}
                    .progress-bar-fill {{
                        background: linear-gradient(to right, #ff6b6b, #ffa700, #ffd400, #a7f000, #6bff00);
                        height: 100%;
                        border-radius: 5px;
                        width: {predicted_percentage}%;
                    }}
                    .progress-text {{
                        font-size: 1.2em;
                        font-weight: bold;
                        color: black   ; /* Use CSS variable */
                        position: absolute;
                        top: 50%;
                        left: {predicted_percentage}%;
                        transform: translate(-50%, -50%);
                    }}
                    .bar-mark {{
                        position: absolute;
                        top: 50%;
                        transform: translateY(-50%);
                        height: 15px;
                        width: 1px;
                        background-color: #ccc;
                        color: var(--bar-mark-text-color); /* Use CSS variable for text */
                        font-size: 0.8em;
                        left: calc(var(--mark-position) * 1%);
                    }}
                    /* Light Theme */
                    .streamlit-light .progress-text {{
                        --progress-text-color: grey;
                    }}
                    .streamlit-light .bar-mark {{
                        --bar-mark-text-color: grey;
                    }}
                    /* Dark Theme */
                    .streamlit-dark .progress-text {{
                        --progress-text-color: black;
                    }}
                    .streamlit-dark .bar-mark {{
                        --bar-mark-text-color: white;
                    }}
                </style>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill">
                        <span class="progress-text">{predicted_percentage}%</span>
                    </div>
                    {''.join([f'<div class="bar-mark" style="--mark-position: {i * 20};">{i * 20}</div>' for i in range(6)])}
                </div>
                <div style="font-size: 1.1em; color: var(--progress-text-color); margin-top: 5px; text-align: center;">{predicted_percentage}%{emoji_for_prediction}</div>
            """, unsafe_allow_html=True)


            st.markdown("### Feature Breakdown")
            for k, v in features.items():
                st.write(f"**{k.replace('_', ' ').title()}**: {round(v, 2)}")

            st.markdown("<h4 style='color: #1e88e5;'>✨ Feature Insights ✨</h4>", unsafe_allow_html=True)
            for k, v in features.items():
                st.markdown(f"<p style='font-size: 1.1em; color: #555;'><strong>{k.replace('_', ' ').title()}</strong>: <span style='color: #007bff;'>{round(v, 2)}</span></p>", unsafe_allow_html=True)


            st.markdown("<h4 style='color: #28a745;'>💬 Communication Patterns 💬</h4>", unsafe_allow_html=True)
            most_common_words, most_common_emojis = analyze_usage(df, selected_user)

            st.markdown("##### Most Frequent Words (excluding common words)")
            if most_common_words:
                for word, count in most_common_words:
                    st.write(f"- <span style='font-weight: bold; color: #6c757d;'>{word}</span>: <span style='color: #00c853;'>{count}</span>", unsafe_allow_html=True)
            else:
                st.write("<span style='color: #6c757d;'>No significant words found after filtering.</span>", unsafe_allow_html=True)

            st.markdown("##### Most Frequent Emojis")
            if most_common_emojis:
                for emo, count in most_common_emojis:
                    st.write(f"- <span style='font-size: 1.5em;'>{emo}</span>: <span style='color: #ffc107;'>{count}</span>", unsafe_allow_html=True)
            else:
                st.write("<span style='color: #6c757d;'>No emojis used by this user.</span>", unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("Error: 'interest_model.pkl' not found. Make sure the model file is in the same directory as the app.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")