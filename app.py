import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pymongo
import streamlit as st
from dotenv import load_dotenv

import utils

# ENV variables
load_dotenv()
aws_access_key_id = os.environ.get('aws_access_key_id')
aws_secret_access_key = os.environ.get('aws_secret_access_key')
region_name = os.environ.get('region_name')
MONGO_URL = os.environ.get('mongo_url')
bucket_name = os.environ.get('bucket_name')
sentiment_key = os.environ.get('sentiment_key')
topic_key = os.environ.get('topic_key')

batch_sentiment_data = utils.read_from_s3(bucket_name, sentiment_key)
batch_sentiment_fig = utils.plot_batch_sentiment_fig(batch_sentiment_data)

stream_sentiment_df = utils.read_from_mongo(MONGO_URL)
stream_sentiment_fig = utils.plot_stream_fig(stream_sentiment_df)

st.plotly_chart(batch_sentiment_fig)
st.plotly_chart(stream_sentiment_fig)
