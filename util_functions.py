import re
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
import streamlit as st
import tqdm


from transformers import pipeline
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")



from keywords import keyword_weights

def parse_popularity(value):
    """
    Converts string like '20 2.6K' or '5.6K' to integer like 5600.
    Picks the highest numeric value in the string.
    """
    numbers = []
    for part in value.split():
        match = re.match(r"([\d.]+)([KkMm]?)", part)
        if match:
            num, suffix = match.groups()
            num = float(num)
            if suffix.lower() == "k":
                num *= 1_000
            elif suffix.lower() == "m":
                num *= 1_000_000
            numbers.append(num)
    return int(max(numbers)) if numbers else 0

def show_insights(filtered_df: pd.DataFrame,resource_related_tweets: pd.DataFrame):
    st.markdown("## 📊 Insights Analysis")
    filtered_df["popularity"] = filtered_df["popularity"].apply(parse_popularity)
    col1, col2 = st.columns(2)
    # --- Most Active Accounts ---
    st.markdown("### 🧑‍💻 Top 5 Most Active Accounts")
    account_counts = filtered_df["account"].value_counts().nlargest(5)
    fig_accounts = px.bar(
        x=account_counts.index,
        y=account_counts.values,
        labels={"x": "Account", "y": "Number of Tweets"},
        title="Top Active Accounts",
        color=account_counts.values,
        color_continuous_scale="Viridis",
        orientation="h"
    )

    st.plotly_chart(fig_accounts,key='fig_accounts')

    # --- Most Popular Tweets ---
    st.markdown("### 🔥 Top 5 Most Popular Informative Tweets")
    top_popular = filtered_df.nlargest(5, "popularity")
    fig_popular = px.bar(
        top_popular,
        x="popularity",
        y="content",
        orientation="h",
        labels={"popularity": "Popularity Score", "content": "Tweet"},
        title="Most Popular Informative Tweets",
        color="popularity",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_popular,key='fig_popular')

    # --- Resource Keywords Frequency ---
    st.markdown("### 📦 Resource Keywords Frequency")
    resource_keywords = [
        "funding", "allocation", "budget", "resources", 
        "investment", "grant", "support", "aid", 
        "donation", "subsidy", "financial assistance", "capital", 
        "sponsorship", "endowment", "fundraising", "resource distribution", 
        "financial support", "public funding", "crowdfunding", "venture capital", 
        "seed funding", "donor", "charity", "philanthropy", "microfinance", 
        "crowdsource", "project funding", "financial aid", "resource mobilization", 
        "capital raise", "business funding", "capital allocation", "debt financing", 
        "equity financing", "grants for", "fund allocation", "financial backing", 
        "monetary support", "subsidized", "investments in", "corporate sponsorship", 
        "relief fund", "charitable fund", "support fund", "budget allocation", 
        "financial resources", "capital investment", "revenue sharing", "resource reallocation", 
        "relief assistance", "donated resources", "revenue funding", "economic support", 
        "aid distribution", "loan assistance", "resource planning", "financial grants", 
        "government funding", "research funding", "budgeting", "monetary aid", 
        "fund disbursement", "financial contribution", "community investment", 
        "relief efforts", "resource allocation strategy", "economic allocation", 
        "financial grants for", "allocation of funds", "capital support", 
        "resources allocation", "economic relief", "wealth distribution", 
        "emergency funds", "resourcing", "resource provision", "fund allocation strategies",
        "food", "ration", "community kitchen", "meals", "free food", "essential supplies"
    ]
    # Count frequency of keywords in tweets
    keyword_counter = Counter()
    for content in filtered_df["content"]:
        content_lower = content.lower()
        for keyword in resource_keywords:
            if keyword.lower() in content_lower:
                keyword_counter[keyword] += 1

    if keyword_counter:
        top_keywords = keyword_counter.most_common(10)
        keywords, counts = zip(*top_keywords)
        fig_keywords = px.bar(
            x=keywords,
            y=counts,
            labels={"x": "Keyword", "y": "Frequency"},
            title="Top Resource Keywords",
            color=counts,
            color_continuous_scale="Viridis"
        )
        with col1:
            st.plotly_chart(fig_keywords,key='fig_keywords 2')
    else:
        st.info("No resource keywords found in the current tweets.")
    

    # Pie Chart: Proportion of resource-related tweets
    st.markdown("### 🥧 Resource Tweet Proportion")
    total_tweets = len(filtered_df)
    resource_tweet_count = len(resource_related_tweets)
    fig_pie = px.pie(
        names=["Resource-related", "Other"],
        values=[resource_tweet_count, total_tweets - resource_tweet_count],
        title="Proportion of Resource-Related Tweets",
        color_discrete_sequence=["red", "lightgrey"]
    )
    with col2:
        st.plotly_chart(fig_pie,key='fig_pie')

    account_counts = filtered_df["account"].value_counts().nlargest(5)

    # --- Pie Chart: Account Contribution Share ---
    st.markdown("### 🥧 Account Contribution Share")

    # Convert the Series to a DataFrame
    account_df = account_counts.reset_index()
    account_df.columns = ["account", "tweet_count"]

    fig5 = px.pie(
        account_df,
        names="account",
        values="tweet_count",
        title="Tweet Contribution by Account",
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig5, use_container_width=True, key="pie_account_contribution")

def score_tweets(text, keyword_weights):
    score = 0
    text_lower = text.lower()
    for keyword, weight in keyword_weights.items():
        if keyword in text_lower:
            score += weight
    return score

def extract_location(text, custom_locations):
    # Run NER model
    ner_results = nlp(text)
    found_locations = [entity['word'] for entity in ner_results if entity['entity'] in ['LOC', 'GPE']]

    # Fallback: keyword match from custom list, ensure custom locations come first
    found_locations_with_custom_first = []
    
    # Add custom locations first
    for loc in custom_locations:
        if loc.lower() in text.lower() and loc not in found_locations_with_custom_first:
            found_locations_with_custom_first.append(loc)
    
    # Add locations found by NER model, excluding any already added custom locations
    for loc in found_locations:
        if loc not in found_locations_with_custom_first:
            found_locations_with_custom_first.append(loc)

    return found_locations_with_custom_first[0] if found_locations_with_custom_first else None


