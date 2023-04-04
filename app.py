import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pymongo
import streamlit as st

import utils

# ENV variables
aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]
region_name = st.secrets["region_name"]
MONGO_URL = st.secrets["mongo_url"]
bucket_name = st.secrets["bucket_name"]
sentiment_key = st.secrets["sentiment_key"]
topic_key = st.secrets["topic_key"]

batch_sentiment_data = utils.read_from_s3(bucket_name, sentiment_key)
batch_sentiment_fig = utils.plot_batch_sentiment_fig(batch_sentiment_data)

stream_sentiment_df = utils.read_from_mongo(MONGO_URL)
stream_sentiment_fig = utils.plot_stream_fig(stream_sentiment_df)

st.plotly_chart(batch_sentiment_fig)
st.plotly_chart(stream_sentiment_fig)
