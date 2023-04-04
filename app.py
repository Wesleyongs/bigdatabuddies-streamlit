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
st.write('This app analyzes sentiment and topics in social media posts about ChatGPT to better understand public perception over time.')

# Explain the data sources and analysis methods
# Explain the data sources and analysis methods
st.markdown("""
    <p>We ingest data from <span style='color:blue; font-weight:bold;'>Twitter</span> and <span style='color:red; font-weight:bold;'>Reddit</span> on the topic of "<span style='color:green; font-weight:bold;'>chatgpt</span>". We use the <span style='color:purple; font-weight:bold;'>VADER</span> sentiment analysis library to perform a time series analysis of sentiment, as well as topic modeling techniques to identify common themes in the data.</p>
    """, unsafe_allow_html=True)
twitter_logo_url = 'https://cdn4.iconfinder.com/data/icons/social-media-icons-the-circle-set/48/twitter_circle-512.png'
reddit_logo_url = 'https://cdn4.iconfinder.com/data/icons/social-messaging-ui-color-shapes-2-free/128/social-reddit-square2-512.png'
gensim_logo_url = 'data:image/png;base64,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'
vader_logo_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRa56yeD_WaLgVxWOHHbuxZTCLjnOQ-3aKuqitphNqwuA&s'
chatgpt_logo_url = 'data:image/png;base64,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'
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
st.plotly_chart(batch_sentiment_fig, responsive=True,
                use_container_width=True, height='100vh')
st.header("Real time sentiment analysis")
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
