import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
import re
from textblob import TextBlob
import emoji
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

st.set_page_config(layout="wide", page_title="Insta Chat Analyzer Pro")

@st.cache_data
def load_chat():
    with open("message_1.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_chat()
messages = data.get('messages', [])
participants = [p['name'] for p in data.get('participants', [])]
user1, user2 = participants[0], participants[1]

msg_count = Counter()
word_count = Counter()
emoji_count = {user1: Counter(), user2: Counter()}
emotional_words = ["love", "miss", "sorry", "heart", "cry", "hug", "happy", "angry", "fight", "hate", "kiss", "care", "feel"]
emotional_counts = {user1: Counter(), user2: Counter()}
sentiments = {user1: [], user2: []}
longest_msg = {"sender": "", "text": "", "words": 0}
daily_msgs = Counter()
first_msg = None
last_msg = None
dry_days = []
prev_date = None

for msg in reversed(messages):
    st.write(msg)  # 👈 This will show you the full structure of each message (first 10)

    sender = msg.get("sender_name", "")
    if sender not in [user1, user2]:
        continue

    text = msg.get("content", "") or msg.get("share", {}).get("link", "") or "sent media"
    ts = msg.get("timestamp_ms")
    if not ts:
        continue

    # Continue your existing logic...

    sender = msg["sender_name"]
    if sender not in [user1, user2]:
        continue

    text = msg.get("content", "")
    ts = msg.get("timestamp_ms")
    dt = datetime.fromtimestamp(ts / 1000)
    date_str = dt.strftime('%Y-%m-%d')
    daily_msgs[date_str] += 1

    msg_count[sender] += 1
    word_count[sender] += len(text.split())

    if not first_msg:
        first_msg = dt
    last_msg = dt

    if len(text.split()) > longest_msg["words"]:
        longest_msg = {"sender": sender, "text": text, "words": len(text.split())}

    for ch in text:
        if ch in emoji.EMOJI_DATA:
            emoji_count[sender][ch] += 1

    for word in emotional_words:
        if word in text.lower():
            emotional_counts[sender][word] += 1

    polarity = TextBlob(text).sentiment.polarity
    sentiments[sender].append(polarity)

    if prev_date:
        gap = (dt.date() - prev_date).days
        if gap > 1:
            dry_days.append(gap)
    prev_date = dt.date()

st.title("💬 Insta Chat Analyzer — Emotional + Timeline Edition")

st.markdown("##### 🤖 Auto Summary")

if first_msg and last_msg:
    st.info(f"📅 Chat started on **{first_msg.strftime('%d %b %Y')}** and ended on **{last_msg.strftime('%d %b %Y')}**")
else:
    st.warning("⚠️ No valid messages found or loaded. Check your JSON or format.")

st.info(f"📝 {user1} sent **{msg_count[user1]}** messages. {user2} sent **{msg_count[user2]}**.")
if longest_msg["words"] > 0:
    st.info(f"💬 Longest message by **{longest_msg['sender']}**: _\"{longest_msg['text'][:100]}...\"_")
if dry_days:
    st.info(f"🥶 Longest dry spell: **{max(dry_days)} days** without messages.")

# PIE CHART SAFE
st.subheader("📊 Message Share")
total_msgs = msg_count[user1] + msg_count[user2]
if total_msgs > 0:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.pie([msg_count[user1], msg_count[user2]],
            labels=[user1, user2],
            autopct="%1.1f%%",
            colors=["#66c2a5", "#fc8d62"])
    st.pyplot(fig1)
else:
    st.warning("No messages to display in pie chart.")

# EMOJIS
st.subheader("😂 Top Emojis")
for user in [user1, user2]:
    top_emojis = emoji_count[user].most_common(5)
    emojis = " ".join([f"{e} ({c})" for e, c in top_emojis]) or "None"
    st.markdown(f"**{user}**: {emojis}")

# EMOTIONAL WORDS
st.subheader("❤️ Emotional Words Used")
for user in [user1, user2]:
    words = ", ".join([f"{w} ({c})" for w, c in emotional_counts[user].most_common(5)]) or "None"
    st.markdown(f"**{user}**: {words}")

# SENTIMENT SAFE
st.subheader("🧠 Avg Sentiment Score")
if sentiments[user1] and sentiments[user2]:
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.barplot(x=[user1, user2], y=[np.mean(sentiments[user1]), np.mean(sentiments[user2])], palette='pastel', ax=ax2)
    ax2.set_ylabel("Sentiment")
    st.pyplot(fig2)
else:
    st.warning("Not enough text to generate sentiment scores.")

# WORD CLOUD SAFE
st.subheader("☁️ Word Cloud")
words_all = []
stop = set(stopwords.words('english'))
for msg in messages:
    if "content" in msg:
        words_all += [w.lower() for w in re.findall(r'\b\w+\b', msg["content"]) if w.lower() not in stop]
if words_all:
    wc = WordCloud(width=800, height=300, background_color="white").generate(" ".join(words_all))
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    ax3.imshow(wc, interpolation="bilinear")
    ax3.axis("off")
    st.pyplot(fig3)
else:
    st.warning("Not enough words for word cloud.")

# HEATMAP SAFE
st.subheader("📆 Daily Message Heatmap")
if daily_msgs:
    msg_df = pd.DataFrame.from_dict(daily_msgs, orient="index", columns=["Messages"])
    msg_df.index = pd.to_datetime(msg_df.index)
    msg_df = msg_df.resample('D').sum().fillna(0)
    msg_df["Week"] = msg_df.index.isocalendar().week
    msg_df["Day"] = msg_df.index.dayofweek
    pivot = msg_df.pivot_table(values="Messages", index="Week", columns="Day", fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.3, ax=ax4, cbar=True)
    st.pyplot(fig4)
else:
    st.warning("No daily message data to build heatmap.")

st.success("✅ Chat analysis complete. App is live and crash-proof 💪")
