import base64
from io import BytesIO
import json
import os

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import pymongo
import streamlit as st
from PIL import Image

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
                        mode='lines', name='Positive', line=dict(color='green'))
    trace2 = go.Scatter(x=df['datetime'], y=df['negative'],
                        mode='lines', name='Negative', line=dict(color='red'))
    trace3 = go.Scatter(x=df['datetime'], y=df['neutral'],
                        mode='lines', name='Neutral', line=dict(color='blue'))

    # Create a data list containing the three trace objects
    data = [trace1, trace2, trace3]

    # Define the layout for the plot
    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Count'))

    # Create a Figure object containing the data and layout, and plot it
    return go.Figure(data=data, layout=layout)


def plot_batch_sentiment_fig(data):
    # Extract the dates and scores from the data
    dates = [d['Date'] for d in data]
    pos_scores = [d['pos'] if d['pos'] is not None else 0 for d in data]
    neg_scores = [d['neg'] if d['neg'] is not None else 0 for d in data]
    neu_scores = [d['neu'] if d['neu'] is not None else 0 for d in data]

    # Create a trace object for the positive scores
    trace1 = go.Bar(x=dates, y=pos_scores, name='Positive',
                    marker=dict(color='green'))

    # Create a trace object for the negative scores, with a base of pos_scores
    trace2 = go.Bar(x=dates, y=neg_scores, name='Negative',
                    marker=dict(color='red'))

    # Create a trace object for the neutral scores, with a base of the sum of pos_scores and neg_scores
    trace3 = go.Bar(x=dates, y=neu_scores, name='Neutral',
                    marker=dict(color='blue'))

    # Create a data list containing the three trace objects
    data = [trace1, trace2, trace3]

    # Define the layout for the plot
    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Compound Score', type='linear',
                   range=[0, 1], dtick=0.1),
        barmode='stack')

    # Create a Figure object containing the data and layout, and plot it

    fig = go.Figure(data=data, layout=layout)
    # fig.show()
    return fig


def show_profile_pictures():

    # Set the path to your image folder
    IMAGE_FOLDER = 'images/'

    # Get a list of all the image filenames in the folder
    image_filenames = os.listdir(IMAGE_FOLDER)

    # Set the number of columns you want to display
    NUM_COLUMNS = 7

    # Calculate the number of rows needed to display all the images
    num_images = len(image_filenames)
    num_rows = num_images // NUM_COLUMNS
    if num_images % NUM_COLUMNS != 0:
        num_rows += 1

    # Create a Streamlit grid to display the images
    for i in range(num_rows):
        # Create a new row
        row = st.columns(NUM_COLUMNS)

        for j in range(NUM_COLUMNS):
            # Calculate the index of the current image in the list
            index = i * NUM_COLUMNS + j

            # Check if there are still images left to display
            if index < num_images:
                # Load the image from file
                image_path = os.path.join(IMAGE_FOLDER, image_filenames[index])
                image = Image.open(image_path)

                # Resize the image to the specified width and height
                image = image.resize((150, 150))

                # Display the image in the current column, with a round border
                with row[j]:
                    st.markdown(
                        f'<figure><img src="data:image/png;base64,{image_to_base64(image)}" style="border-radius: 50%; overflow: hidden;" /><figcaption>{image_filenames[index][:-4]}</figcaption></figure><br>', unsafe_allow_html=True)
                    # st.caption(
                    #     f'<div style="text-align: left;">{image_filenames[index]}</div>', unsafe_allow_html=True)


# Function to convert image to base64 string


def image_to_base64(image):
    with BytesIO() as buffer:
        image.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()
