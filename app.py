import streamlit as st
import json
from collections import Counter
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def load_messages(data):
    messages = []
    try:
        # Try all known formats
        if "messages" in data:
            raw = data["messages"]
        elif "chat" in data and "messages" in data["chat"]:
            raw = data["chat"]["messages"]
        else:
            st.error("No valid messages key found.")
            return []

        for msg in raw:
            sender = msg.get("sender_name", "Unknown")
            content = msg.get("content") or msg.get("text") or msg.get("message") or ""
            timestamp = msg.get("timestamp_ms") or msg.get("timestamp")
            if content and isinstance(content, str):
                messages.append({
                    "sender": sender,
                    "message": content,
                    "timestamp": timestamp
                })
        return messages

    except Exception as e:
        st.error(f"Error loading chat: {e}")
        return []

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

st.set_page_config(page_title="Chat Analyzer", layout="wide")
st.title("📊 Allatyzer – Instagram Chat Emotional Analyzer")

uploaded_file = st.file_uploader("Upload your Instagram chat JSON", type="json")

if uploaded_file:
    try:
        data = json.load(uploaded_file)
        messages = load_messages(data)

        if len(messages) == 0:
            st.warning("No valid messages found.")
        else:
            df = pd.DataFrame(messages)
            st.success(f"Loaded {len(df)} messages from {df['sender'].nunique()} participants.")

            # Pie Chart
            sender_counts = df["sender"].value_counts()
            fig = px.pie(values=sender_counts.values, names=sender_counts.index, title="Message Share")
            st.plotly_chart(fig)

            # Word Cloud
            st.subheader("Word Cloud")
            generate_wordcloud(df["message"].tolist())

            # Sentiment Analysis
            sia = SentimentIntensityAnalyzer()
            df["sentiment"] = df["message"].apply(lambda x: sia.polarity_scores(x)["compound"])
            avg_sentiment = df["sentiment"].mean()
            st.metric(label="Average Sentiment Score", value=round(avg_sentiment, 3))

            # Heatmap
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["day"] = df["time"].dt.date
            daily_count = df.groupby("day").count()["message"].reset_index(name="messages")
            st.subheader("Daily Message Heatmap")
            fig2 = px.density_heatmap(daily_count, x="day", y="messages", nbinsx=30, color_continuous_scale="Blues")
            st.plotly_chart(fig2)

    except json.decoder.JSONDecodeError:
        st.error("File is not a valid JSON.")