import pandas as pd
import requests
import time
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RedditAPI:
    """
    Manages interactions with the Reddit API, including fetching stories, saving to CSV,
    and handling rate limits.

    Attributes:
        client_id (str): The client ID for Reddit API access.
        client_secret (str): The client secret for Reddit API access.
        user_agent (str): The user agent string for Reddit API requests.
        access_token (str): The current access token for OAuth.
        token_expiry (float): The expiry timestamp of the current access token.
        request_count_file (str): File path to store request count data.
        request_count (int): Number of API requests made in the current period.
    """
    def __init__(self, client_id, client_secret, user_agent='MyRedditApp/0.1'):
        """
        Initializes the RedditAPI object with client credentials and user agent.

        Args:
            client_id (str): The client ID for Reddit API access.
            client_secret (str): The client secret for Reddit API access.
            user_agent (str): The user agent string for Reddit API requests, default is 'MyRedditApp/0.1'.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.access_token = None
        self.token_expiry = None
        self.request_count_file = "../../.temp/request_count.json"
        self.request_count = 0
        self.load_request_count()

    def get_new_access_token(self):
        """
        Obtains a new access token from Reddit API using client credentials.
        
        Raises:
            Exception: If the request for a new access token fails.
        """
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        data = {'grant_type': 'client_credentials'}
        headers = {'User-Agent': self.user_agent}
        response = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)

        if response.status_code == 200:
            token_data = response.json()
            logging.info("Access token obtained")
            self.access_token = token_data['access_token']
            self.token_expiry = time.time() + token_data['expires_in']
        else:
            logging.error(f"Failed to get access token, status code: {response.status_code}")
            raise Exception(f"Failed to get access token, status code: {response.status_code}")

    def make_request(self, endpoint):
        """
        Makes a request to the Reddit API at the specified endpoint.

        Args:
            endpoint (str): The API endpoint to request.

        Returns:
            dict: The JSON response from the API request.

        Raises:
            Exception: If the API request fails.
        """
        if self.access_token is None or time.time() >= self.token_expiry:
            self.get_new_access_token()

        headers = {
            'Authorization': f'bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        response = requests.get(f'https://oauth.reddit.com{endpoint}', headers=headers)

        if response.status_code == 200:
            logging.info(f"Request successful for {endpoint}")
            return response.json()
        else:
            logging.error(f"API request failed for {endpoint}, status code: {response.status_code}")
            raise Exception(f"API request failed, status code: {response.status_code}")
        
    def load_request_count(self):
        """
        Loads the request count from file, resetting if last run was over a minute ago.
        """
        if os.path.exists(self.request_count_file):
            with open(self.request_count_file, 'r') as file:
                data = json.load(file)
                last_run = data.get('last_run', 0)
                if time.time() - last_run > 60:  # More than 1 minute ago
                    self.request_count = 0
                else:
                    self.request_count = data.get('request_count', 0)
        else:
            self.request_count = 0

    def update_request_count(self):
        """
        Updates and saves the current request count to file.
        """
        directory = os.path.dirname(self.request_count_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(self.request_count_file, 'w') as file:
            data = {
                'request_count': self.request_count,
                'last_run': time.time()
            }
            json.dump(data, file)

    def save_stories_to_csv(self, stories, file_path):
        """
        Saves the fetched stories to a CSV file at the specified path.

        Args:
            stories (list of dict): The stories to save.
            file_path (str): The file path to save the stories CSV.
        """
        df = pd.DataFrame(stories)
        df.to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path), index=False)

    def check_rate_limit(self, sleep_time=61):
        """
        Checks if the request count is approaching the rate limit and pauses if necessary.
        """
        if self.request_count >= 100:
            logging.info(f"Approaching rate limit, pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)
            self.request_count = 0

    def process_post(self, post, existing_data, subreddit, score_limit):
        """
        Processes a single post, filtering based on ID and score.

        Args:
            post (dict): The post data to process.
            existing_data (list): List of existing post IDs to avoid duplicates.
            subreddit (str): The subreddit from which the post was fetched.

        Returns:
            dict or None: Processed post data or None if post is filtered out.
        """
        post_id = post['data']['id']
        if post_id in existing_data or post['data']['score'] < score_limit:
            if post['data']['score'] < score_limit:
                logging.debug(f"Skipping post {post_id} from {subreddit}, score: {post['data']['score']}")
                pass
            return None
        
        logging.debug(f"Processing post {post_id} from {subreddit}")
        return {
            'subreddit': subreddit,
            'post_id': post_id,
            'title': post['data']['title'],
            'selftext': post['data']['selftext'],
            'url': post['data']['url'],
            'score': post['data']['score'],
            'media': post['data'].get('media'),
            'is_video': post['data']['is_video']
        }

    def fetch_data_for_subreddit(self, subreddit, limit, existing_data, score_limit, type, time):
        """
        Fetches top stories for a subreddit within the specified limit.

        Args:
            subreddit (str): The subreddit to fetch stories from.
            limit (int): The maximum number of stories to fetch.
            existing_data (list): List of existing post IDs to avoid duplicates.

        Returns:
            list of dict: A list of fetched and processed stories.
        """
        logging.info(f'Starting to fetch data for subreddit: {subreddit} with limit: {limit}')
        stories = []
        endpoint = f'/r/{subreddit}/{type}?limit={limit}&t={time}'
        response_data = self.make_request(endpoint)
        if response_data:
            logging.info(f'Successfully fetched data for subreddit: {subreddit}')
        else:
            logging.error(f'Failed to fetch data for subreddit: {subreddit}')
            return stories  # Early return in case of failure

        for post in response_data['data']['children']:
            story = self.process_post(post, existing_data, subreddit, score_limit)
            if story:
                stories.append(story)
        
        logging.info(f'Finished fetching and processing data for subreddit: {subreddit}. Total stories fetched: {len(stories)}')
        return stories

    def load_existing_data(self, file_path):
        """
        Loads existing data from a CSV file to avoid duplicating posts.

        Args:
            file_path (str): The file path from which to load existing data.

        Returns:
            list: A list of post IDs from the existing data.
        """
        return pd.read_csv(file_path)['post_id'].tolist() if pd.io.common.file_exists(file_path) else []

    def fetch_stories(self, subreddits, limit=100, score_limit=100, file_path="reddit_stories.csv", type='new', time='all'):
        """
        Fetches and saves stories from specified subreddits to a CSV file.

        Args:
            subreddits (list of str): Subreddits from which to fetch stories.
            limit (int): The maximum number of stories to fetch per subreddit.
            file_path (str): The file path to save fetched stories.
        """
        existing_data = self.load_existing_data(file_path)
        for subreddit in subreddits:
            self.check_rate_limit(sleep_time=61)
            self.request_count += 1
            self.update_request_count()
            try:
                stories = self.fetch_data_for_subreddit(subreddit, limit, existing_data, score_limit, type, time)
                if stories:
                    self.save_stories_to_csv(stories, file_path)
            except Exception as e:
                logging.error(f"Error fetching from {subreddit}: {e}")
