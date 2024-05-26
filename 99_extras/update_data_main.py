import os
import subprocess
import logging
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from data_generator.reddit.utils.reddit_api_caller import RedditAPI

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_api = RedditAPI(client_id, client_secret)
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'

# Specify your cache directory path
cache_dir = "/mnt/f/HuggingFace/huggingface_cache"

collection_name = "reddit_stories"
dim = 384  # Dimension of embeddings

# Filepath for the Reddit stories CSV
file_path = "data/reddit/reddit_stories.csv"

subreddits = [
    "LetsNotMeet",
    "UnresolvedMysteries",
    "Relationships",
    "AmItheAsshole",
    "Relationship_Advice",
    "OnlineDating",
    "Paranormal"
]

# Function to start Milvus
def start_milvus(start_script="./vector_database/standalone_embed.sh start"):
    # Start Milvus
    subprocess.run(start_script, shell=True, check=True)
    logging.info("Milvus started successfully.")

# Function to check if Milvus is running and connect
def connect_to_milvus(host='localhost', port='19530'):
    try:
        connections.connect("default", host=host, port=port)
        logging.info("Connected to Milvus successfully.")
    except Exception as e:
        logging.error("Failed to connect to Milvus. Attempting to start Milvus...")
        start_milvus()
        # Wait a moment for Milvus to fully start
        time.sleep(20)
        connections.connect("default", host=host, port=port)
        logging.info("Connected to Milvus after starting the service.")

# Connect to Milvus
connect_to_milvus(host=MILVUS_HOST, port=MILVUS_PORT)

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

### Upload new content to milvus (vector database)
    
# Function to delete and recreate the collection
def recreate_collection(name, dim):
    # Check if the collection exists
    if utility.has_collection(name):
        # Delete the existing collection
        utility.drop_collection(name)
        logging.info(f"Collection {name} deleted.")

    # Create a new collection
    fields = [
        FieldSchema(name="post_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="subreddit", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),  # Adjust length as needed
        FieldSchema(name="selftext", dtype=DataType.VARCHAR, max_length=65535),  # Adjust length as needed
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="score", dtype=DataType.INT64),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Reddit stories embeddings")
    collection = Collection(name=name, schema=schema)
    logging.info(f"Collection {name} created.")
    return collection

# Adjusted main process to delete and recreate the collection
collection = recreate_collection(collection_name, dim)

# Load the stories from the CSV file
def load_reddit_stories(file_path):
    return pd.read_csv(file_path)

df_stories = load_reddit_stories(file_path)

# Initialize the Sentence Transformer model with a cache directory
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)

# Generate embeddings for the textual content
def generate_embeddings(df):
    texts = (df['title'] + " " + df['selftext']).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

embeddings = generate_embeddings(df_stories)

# Function to insert data into the Milvus collection with adjustments for string conversion
def insert_data_into_milvus(collection, df, embeddings):
    # Convert all DataFrame columns to string to ensure compatibility with Milvus
    df = df.astype(str)  # Convert entire DataFrame to string type
    df.score = df.score.astype(int)  # Convert score to integer type
    
    # Print number of rows with NaN values
    logging.info(f"Number of rows with NaN values: {df.isna().sum().sum()}")
    
    # Prepare the fields according to the collection schema
    entities = [
        df["post_id"].tolist(),
        df["subreddit"].tolist(),
        df["title"].tolist(),
        df["selftext"].tolist(),
        df["url"].tolist(),
        df["score"].tolist(),
        embeddings.tolist(),
    ]
    
    # Insert the data into the collection
    mr = collection.insert(entities)
    collection.load()  # Load the collection into memory to make it searchable
    logging.info(f"Inserted {len(df)} stories into the collection '{collection.name}'.")


# Function to create an index on the embeddings field
def create_index_for_collection(collection, dim):
    index_params = {
        "index_type": "IVF_FLAT",  # Choose an index type suitable for your needs
        "metric_type": "L2",  # Use "L2" for Euclidean distance, "IP" for inner product, etc.
        "params": {"nlist": 24},  # Adjust based on your dataset size and search requirements
    }
    collection.create_index(field_name="embeddings", index_params=index_params)
    logging.info("Index created for the 'embeddings' field.")

# Call the function to create an index after the collection is created
create_index_for_collection(collection, dim)

# Call the function to insert your data
insert_data_into_milvus(collection, df_stories, embeddings)

def search_similar_stories(query_text, top_k, collection, model):
    # Generate the embedding for the query text
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    
    # Ensure the query_embedding is in the correct format: a list of lists
    query_embedding = query_embedding.tolist()  # Convert NumPy array to list
    if not isinstance(query_embedding[0], list):
        # Make sure it's a list of lists even for a single query
        query_embedding = [query_embedding]

    # Define search parameters
    search_params = {
        "metric_type": "L2",  # or "IP" for inner product
        "params": {"nprobe": 24},  # Adjust based on your index and search requirements
    }
    
    # Perform the search
    results = collection.search(
        data=query_embedding, 
        anns_field="embeddings", 
        param=search_params, 
        limit=top_k, 
        output_fields=["post_id", "title", "subreddit", "url"]  # Adjust based on the fields you want to retrieve
    )
    
    # Process and return the search results
    for hits in results:
        for hit in hits:
            print(f"Post ID: {hit.entity.get('post_id')}, Title: {hit.entity.get('title')}, Subreddit: {hit.entity.get('subreddit')}, URL: {hit.entity.get('url')}, Distance: {hit.distance}")
            
    return results


# Example usage
query_text = "scary story about a ghost"
top_k = 5  # Number of top similar stories you want to retrieve
search_results = search_similar_stories(query_text, top_k, collection, model=model)
