import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pymongo
import streamlit as st

import utils

# configs
st.set_page_config(page_title="My App", layout="wide")

# ENV variables
aws_access_key_id = st.secrets.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets.secrets["aws_secret_access_key"]
region_name = st.secrets.secrets["region_name"]
MONGO_URL = st.secrets.secrets["mongo_url"]
bucket_name = st.secrets.secrets["bucket_name"]
sentiment_key = st.secrets.secrets["sentiment_key"]
topic_key = st.secrets.secrets["topic_key"]

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
