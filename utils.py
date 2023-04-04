import boto3
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import pymongo
import json
import streamlit as st

# ENV variables
aws_access_key_id = st.secrets.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets.secrets["aws_secret_access_key"]
region_name = st.secrets.secrets["region_name"]


def read_from_s3(bucket_name, key):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    s3 = session.resource('s3')

    obj = s3.Object(bucket_name, key)
    print(aws_access_key_id)
    print(aws_secret_access_key)
    print(region_name)
    print(bucket_name)
    print(aws_secret_access_key)
    json_data = obj.get()['Body'].read().decode('utf-8')
    return json.loads(json_data)


def read_from_mongo(MONGO_URL):
    # set up the MongoDB client
    client = pymongo.MongoClient(
        MONGO_URL
    )

    # access the "my_database" database and the "my_collection" collection
    db = client.Tweets
    collection = db["real-time"]

    # Read data from MongoDB
    data = []
    for document in collection.find():
        date = document["_id"]
        sentiment_data = document["sentiment_data"]
        for time, sentiment_values in sentiment_data.items():
            positive_count = sentiment_values["positive_count"]
            negative_count = sentiment_values["negative_count"]
            neutral_count = sentiment_values["neutral_count"]
            data.append((date, time, float(positive_count),
                        int(negative_count), float(neutral_count)))

    # Convert data to pandas DataFrame
    df = pd.DataFrame(
        data, columns=["date", "time", "positive", "negative", "neutral"])
    df.drop(columns=['date'], inplace=True)
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values(by='datetime')


def plot_stream_fig(df):
    # Create three trace objects, one for each score type
    trace1 = go.Scatter(x=df['datetime'], y=df['positive'],
                        mode='lines', name='Positive')
    trace2 = go.Scatter(x=df['datetime'], y=df['negative'],
                        mode='lines', name='Negative')
    trace3 = go.Scatter(x=df['datetime'], y=df['neutral'],
                        mode='lines', name='Neutral')

    # Create a data list containing the three trace objects
    data = [trace1, trace2, trace3]

    # Define the layout for the plot
    layout = go.Layout(title='Sentiment Scores Over Time',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Score'))

    # Create a Figure object containing the data and layout, and plot it
    return go.Figure(data=data, layout=layout)


def plot_batch_sentiment_fig(data):
    # Extract the dates and scores from the data
    dates = [d['Date'] for d in data]
    pos_scores = [d['pos'] if d['pos'] is not None else 0 for d in data]
    neg_scores = [d['neg'] if d['neg'] is not None else 0 for d in data]
    neu_scores = [d['neu'] if d['neu'] is not None else 0 for d in data]

    # Create a trace object for the positive scores
    trace1 = go.Bar(x=dates, y=pos_scores, name='Positive')

    # Create a trace object for the negative scores, with a base of pos_scores
    trace2 = go.Bar(x=dates, y=neg_scores, name='Negative', base=pos_scores)

    # Create a trace object for the neutral scores, with a base of the sum of pos_scores and neg_scores
    trace3 = go.Bar(x=dates, y=neu_scores, name='Neutral', base=[
                    sum(x) for x in zip(pos_scores, neg_scores)])

    # Create a data list containing the three trace objects
    data = [trace1, trace2, trace3]

    # Define the layout for the plot
    layout = go.Layout(title='Sentiment Scores Over Time',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Score', type='linear',
                                  range=[0, 1], dtick=0.1),
                       barmode='stack')

    # Create a Figure object containing the data and layout, and plot it

    fig = go.Figure(data=data, layout=layout)
    # fig.show()
    return fig
