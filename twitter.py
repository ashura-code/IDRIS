from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



import os
import pickle
import json
import time
import pandas as pd
from tqdm import tqdm
import re


from Models.InformationClassifier import classify_tweet_info_api, classify_tweet_info_custom
from Models.ResourceFinder import get_resource_tweets


tqdm.pandas()
# Path to your local chromedriver
chromedriver_path = "/Users/adhitya/Desktop/IDRIS/chromedriver-mac-arm64/chromedriver"



def get_latest_tweets(username,chromedriver_path):
        """
        Get the latest tweets from a given username using Selenium.
        """

        service = Service(chromedriver_path)
        options = Options()

        driver = webdriver.Chrome(service=service, options=options)

        driver.get(f'https://x.com/{username}') 

        with open('cookies.json', 'r') as cookie_file:
            cookies = json.load(cookie_file)

        for cookie in cookies:
            # Remove or modify the 'SameSite' attribute
            if 'sameSite' in cookie:
                cookie['sameSite'] = 'Lax'  # Change to 'Strict' or 'None' if needed
            else:
                cookie['sameSite'] = 'Lax'  # Set a default 'SameSite' if not present

            # Some cookies might require setting the 'domain' before adding
            if 'expiry' in cookie:
                del cookie['expiry']  # Optional: Remove expiry to prevent potential issues
            
            driver.add_cookie(cookie)

        driver.refresh()

        # Do your automation tasks here
        time.sleep(5)  # Wait for login to complete

        # Scroll and collect HTML
        all_html = ''
        unique_tweets = set()
        end_time = time.time() + 15
        print("Scrolling and collecting tweets...")
        while time.time() < end_time:
            html_chunk = driver.page_source
            soup = BeautifulSoup(html_chunk, 'html.parser')
            articles = soup.find_all('article', attrs={'data-testid': 'tweet'})

            
            for article in articles:
                text = article.get_text(separator=' ', strip=True)
                if text not in unique_tweets:
                    unique_tweets.add(text)
                    # print("======================")
                    # print(text)
                    # print("======================")

            driver.execute_script("window.scrollBy(0, 5000)")
            time.sleep(2)
        return unique_tweets

def tweets_to_df(tweets: set) -> pd.DataFrame:
    """
    Convert a set of tweets to a DataFrame.
    """
    records = []

    for tweet in tweets:
        try:
            if '·' not in tweet:
                continue
            account, rest = tweet.split('·', 1)
            popularity_match = re.search(r'(\d+(?:\.\d+[KMB]?)?(?:\s+\d+(?:\.\d+[KMB]?))*)$', rest)
            if popularity_match:
                popularity = popularity_match.group(0).strip()
                content = rest.replace(popularity, '').strip()
            else:
                popularity = ''
                content = rest.strip()
            records.append({
                'account': account.strip(),
                'content': content,
                'popularity': popularity
            })
        except Exception as e:
            print(f"Error processing tweet: {e}")
            continue
    # Create DataFrame
    return pd.DataFrame(records)

def filter_tweets(df:pd.DataFrame) -> pd.DataFrame:
    """
    Filter tweets based on if its informative or not.
    """
    def classify_row_safe(text):
        try:
            return classify_tweet_info_custom(text)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            return None  # or use a special label like "error"
    df["predicted_label"] = df["content"].progress_apply(classify_row_safe)
    filtered_df = df[df["predicted_label"] == "informative"][["account", "content","popularity"]].reset_index(drop=True)
    return filtered_df

def get_resource_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get tweets that contain resource allocation tweets.
    """
    return get_resource_tweets(df)
    


# if __name__ == "__main__":
#     # Example usage
#     # Ensure you have a valid chromedriver path and cookies.json file
#     username = "NDRFHQ"
#     tweets = get_latest_tweets(username,chromedriver_path)
#     df = tweets_to_df(tweets)
#     filtered_df = filter_tweets(df)
#     print(filtered_df.head())
#     print(filtered_df.shape)
#     food_df = get_resource_df(filtered_df)
#     print(food_df.head())

