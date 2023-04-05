import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pymongo
import streamlit as st

import utils

# ENV variables
aws_access_key_id = st.secrets.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets.secrets["aws_secret_access_key"]
region_name = st.secrets.secrets["region_name"]
MONGO_URL = st.secrets.secrets["mongo_url"]
bucket_name = st.secrets.secrets["bucket_name"]
sentiment_key = st.secrets.secrets["sentiment_key"]
topic_key = st.secrets.secrets["topic_key"]

# Define the page title and an introduction message
st.set_page_config(page_title='ChatGPT Sentiment Analysis',
                   page_icon=':bar_chart:', layout='wide')
st.title('Welcome to the ChatGPT Sentiment Analysis App')
utils.show_profile_pictures()
st.write('This app analyzes sentiment and topics in social media posts about ChatGPT to better understand public perception over time.')

# Explain the data sources and analysis methods
# Explain the data sources and analysis methods
st.markdown("""
    <p>We ingest data from <span style='color:blue; font-weight:bold;'>Twitter</span> and <span style='color:red; font-weight:bold;'>Reddit</span> on the topic of "<span style='color:green; font-weight:bold;'>chatgpt</span>". We use the <span style='color:purple; font-weight:bold;'>VADER</span> sentiment analysis library to perform a time series analysis of sentiment, as well as topic modeling techniques to identify common themes in the data.</p>
    """, unsafe_allow_html=True)
twitter_logo_url = 'https://cdn4.iconfinder.com/data/icons/social-media-icons-the-circle-set/48/twitter_circle-512.png'
reddit_logo_url = 'https://cdn4.iconfinder.com/data/icons/social-messaging-ui-color-shapes-2-free/128/social-reddit-square2-512.png'
gensim_logo_url = 'https://repository-images.githubusercontent.com/1349775/202c4680-8f7c-11e9-91c6-745fdcbeffe8'
vader_logo_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRa56yeD_WaLgVxWOHHbuxZTCLjnOQ-3aKuqitphNqwuA&s'
chatgpt_logo_url = 'https://cdn2.iconfinder.com/data/icons/artificial-intelligence-ai/64/openai-gym-Toolkit-algorithm-Reinforcement-Learning_-256.png'
cola, colb, colc = st.columns(3)
cola.write("These are our data sources:")
cola.image(twitter_logo_url, width=50)
cola.image(reddit_logo_url, width=50)
colb.write("These are our topics:")
colb.image(chatgpt_logo_url, width=50)
colc.write("These are our ML libraries:")
colc.image(vader_logo_url, width=50)
colc.image(gensim_logo_url, width=50)

# Explain the app's goal and how it can be used
st.write('Our goal is to help users better understand public perception toward ChatGPT over time. By analyzing sentiment and topics in social media posts, we hope to provide insights that can inform strategic decisions related to ChatGPT development and marketing. You can use this app to explore the sentiment and topics related to ChatGPT, and to stay up to date on public perception over time.')

# Provide links to data sources and further reading
st.write('For more information about the data sources and analysis methods, please see the links below:')
st.markdown('- [Twitter API documentation](https://developer.twitter.com/en/docs)\n- [Reddit API documentation](https://www.reddit.com/dev/api/)\n- [VADER sentiment analysis library](https://github.com/cjhutto/vaderSentiment)\n- [Topic modeling techniques](https://en.wikipedia.org/wiki/Topic_model)')

batch_sentiment_data = utils.read_from_s3(bucket_name, sentiment_key)
batch_sentiment_fig = utils.plot_batch_sentiment_fig(batch_sentiment_data)

stream_sentiment_df = utils.read_from_mongo(MONGO_URL)
stream_sentiment_fig = utils.plot_stream_fig(stream_sentiment_df)
    
st.header("Sentiment distribution over time")
st.write("### What is Compound Score?")
st.write("In sentiment analysis, the compound score is a metric that summarizes the overall sentiment of a piece of text, such as a tweet or a product review. The score ranges from -1 (most extreme negative sentiment) to 1 (most extreme positive sentiment), with 0 representing neutral sentiment.")
st.write("The compound score corresponds to the sum of all the valence scores normalized between -1 and 1, where the valence score is a measure of the positivity or negativity of a word.")
st.write("To learn more about compound score and how it's calculated, check out this article on social media sentiment analysis with VADER:")
st.write("[Social Media Sentiment Analysis in Python with VADER: No Training Required](https://towardsdatascience.com/social-media-sentiment-analysis-in-python-with-vader-no-training-required-4bc6a21e87b8)")
st.plotly_chart(batch_sentiment_fig, responsive=True,
                use_container_width=True, height='100vh')
st.header("Real time sentiment analysis")
st.write("Straight lines are for when the streaming service is not running (to save monies)")
st.plotly_chart(stream_sentiment_fig, responsive=True,
                use_container_width=True, height='100vh')

st.header("Frequently occuring keywords accross the various sentiment groups")
batch_topic_data = utils.read_from_s3(bucket_name, topic_key)
# Define a dictionary with colors for each sentiment
sentiment_colors = {
    'positive': 'green',
    'neutral': 'blue',
    'negative': 'red'
}

# Create three equal-sized columns
col1, col2, col3 = st.columns(3)

# Loop over the sentiments and display each in a separate column
for i, (sentiment, values) in enumerate(batch_topic_data.items()):
    # Select the appropriate column based on the index
    if i % 3 == 0:
        col = col1
    elif i % 3 == 1:
        col = col2
    else:
        col = col3

    # Display the sentiment in the selected column, with color
    col.subheader(f":{sentiment_colors[sentiment]}[{sentiment.upper()}]")
    for key, words in values.items():
        col.write(f'Topic {key}: {", ".join(words)}')

# Read the JSON data from file
st.header("Frequently occuring keywords across the months")
with open('monthly_topic.json', 'r') as f:
    json_data = json.load(f)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.subheader("November")
for key, words in json_data["November"].items():
        col1.write(f'{key}: {", ".join(words)}')
col2.subheader("December")
for key, words in json_data["December"].items():
        col2.write(f'{key}: {", ".join(words)}')
col3.subheader("January")
for key, words in json_data["January"].items():
        col3.write(f'{key}: {", ".join(words)}')
col4.subheader("Febuary")
for key, words in json_data["Febuary"].items():
        col4.write(f'{key}: {", ".join(words)}')
col5.subheader("March")
for key, words in json_data["March"].items():
        col5.write(f'{key}: {", ".join(words)}')
col6.subheader("April")
for key, words in json_data["April"].items():
        col6.write(f'{key}: {", ".join(words)}')