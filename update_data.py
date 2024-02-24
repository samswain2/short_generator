import os
import logging
from dotenv import load_dotenv

from data_generator.reddit.utils.reddit_api_caller import RedditAPI

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_api = RedditAPI(client_id, client_secret)

subreddits = [
    "LetsNotMeet",
    "UnresolvedMysteries",
    "Relationships",
    "AmItheAsshole",
    "Relationship_Advice",
    "OnlineDating",
    "Paranormal"
]

for type in ["new", "top"]:
    logging.info(f"Fetching {type} stories from Reddit")
    reddit_api.fetch_stories(
        subreddits=subreddits, 
        limit=1000, 
        score_limit=100, 
        file_path="data/reddit/reddit_stories.csv",
        type=type, 
        time='all'
    )