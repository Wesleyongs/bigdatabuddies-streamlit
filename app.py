import pymongo
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

# set up the MongoDB client
client = pymongo.MongoClient(
    "mongodb+srv://bda:B1gdata2023@cluster0.igmsvhv.mongodb.net/Tweets?retryWrites=true&w=majority")

# access the "my_database" database and the "my_collection" collection
db = client.Tweets
collection = db["real-time"]

# Read data from MongoDB
data = []
for document in collection.find():
    date = document["_id"]
    sentiment_data = document["sentiment_data"]
    for time, sentiment_values in sentiment_data.items():
        compound = sentiment_values["compound"]
        count = sentiment_values["count"]
        total = sentiment_values["total"]
        data.append((date, time, float(compound), int(count), float(total)))

# Convert data to pandas DataFrame
df = pd.DataFrame(data, columns=["date", "time", "compound", "count", "total"])
df.drop(columns=['date'], inplace=True)
df.rename(columns={'time': 'datetime'}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')

# Plot time series of compound sentiment
fig = px.line(df, x='datetime', y='compound', title='Compound Sentiment Time Series')
fig.update_xaxes(title='Datetime')
fig.update_yaxes(title='Compound Sentiment')

# Display the plot on Streamlit
st.plotly_chart(fig)
