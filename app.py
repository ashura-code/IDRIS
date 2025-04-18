import requests
import tqdm

#Custom Imports
from twitter import get_latest_tweets, tweets_to_df, filter_tweets, get_resource_df;
from util_functions import parse_popularity, show_insights, score_tweets ,extract_location
from keywords import keyword_weights,custom_locations
chromedriver_path = "/Users/adhitya/Desktop/IDRIS/chromedriver-mac-arm64/chromedriver"

#Other Essential Imports
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
import streamlit as st

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700&display=swap');

    .header-container {
        height: 3vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
    }

    .header-title {
        font-family: 'Outfit', sans-serif;
        font-size: 4rem;
        font-stretch: extra-expanded;
        letter-spacing: 0.1em;
        color: black;
        margin: 0;
    }
    </style>

    <div class="header-container">
        <h1 class="header-title">IRIS</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Remove default padding from main container */
    .block-container {
        padding-left: 0rem;
        padding-right: 0rem;
    }

    /* Make columns take full width */
    .stColumns {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    .stColumn {
        padding: 0 !important;
        flex: 1 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


user_channel = st.text_area("Enter X channel", "")


if st.button("🔍 Fetch & Analyze Tweets"):
    with st.spinner("Fetching the latest tweets.."):
     try:
        tweets = get_latest_tweets(user_channel, chromedriver_path)
        st.success("✅ Tweets Collected!")

        st.write("🔍 Analyzing tweets...")

        with st.expander("✅ Collected Tweets:"):
            df = tweets_to_df(tweets)
            st.dataframe(df,use_container_width=True)

        with st.expander("✅Informative Tweets:"):
            filtered_df = filter_tweets(df)
            filtered_df['location'] = filtered_df['content'].apply(lambda x: extract_location(x,custom_locations))
            st.dataframe(filtered_df,use_container_width=True)

        with st.expander("Tweets based on criticalness"):
            critical_df = filtered_df
            critical_df['critical'] = critical_df["content"].apply(lambda x: score_tweets(x, keyword_weights))
            df_sorted = critical_df.sort_values(by="critical", ascending=False)
            st.write("Critical Tweets:")
            st.dataframe(df_sorted,use_container_width=True)



        with st.expander("🚨 Tweets asking for Resources:"):
            resource_related_tweets=get_resource_df(filtered_df)
            st.dataframe(resource_related_tweets)

        show_insights(filtered_df=filtered_df, resource_related_tweets=resource_related_tweets)


     except Exception as e:
        st.error(f"Uhh an error occured here;s the error: {e}")
