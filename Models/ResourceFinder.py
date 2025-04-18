from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

resource_keywords = [
    "funding", "allocation", "budget", "resources", 
    "investment", "grant", "support", "aid", 
    "donation", "subsidy", "financial assistance", "capital",
    "sponsorship", "endowment", "fundraising", "resource distribution",
    "financial support", "public funding", "crowdfunding", "venture capital", 
    "seed funding", "donor", "charity", "philanthropy", "microfinance", 
    "crowdsource", "project funding", "financial aid", "resource mobilization","food"
]



# Function to classify tweet based on similarity to resource-related keywords
def classify_tweet(tweet, vectorizer, keywords_tfidf):
    # Transform the tweet into TF-IDF vector
    tweet_tfidf = vectorizer.transform([tweet])
    
    # Compute cosine similarities between the tweet and the keyword embeddings
    similarities = cosine_similarity(tweet_tfidf, keywords_tfidf)
    
    # If the max similarity is above a certain threshold, classify as "resource allocation"
    max_similarity = similarities.max()
    if max_similarity > 0.3:  # Adjust threshold as needed
        return "Resource Allocation"
    else:
        return "Other"

    
def get_resource_tweets(df: pd.DataFrame) -> pd.DataFrame:

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit the vectorizer on resource keywords
    keywords_tfidf = vectorizer.fit_transform(resource_keywords)
    
    # Classify tweets
    def classify_row(text):
        return classify_tweet(text, vectorizer, keywords_tfidf)
    
    # Apply classification to each tweet
    df["predicted_label"] = df["content"].apply(classify_row)
    
    # Filter tweets that are classified as "Resource Allocation"
    resource_tweets = df[df["predicted_label"] == "Resource Allocation"]
    return resource_tweets


# if __name__ == "__main__":
#     # Example dataframe with tweet content
#     data = {
#         "account": ["user1", "user2", "user3"],
#         "content": [
#             "We are looking for funding to support our community project.",
#             "Join us for a webinar on cyber security.",
#             "The government has announced new resource allocation for health."
#         ],
#         "popularity": [100, 200, 300]
#     }
#     df = pd.DataFrame(data)

#     # Get resource allocation tweets
#     resource_tweets = get_resource_tweets(df)
#     print(resource_tweets)