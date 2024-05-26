import logging
import os
import subprocess
import time

from pymilvus import (
    connections, Collection,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to start Milvus
def start_milvus(start_script="./vector_database/standalone_embed.sh start"):
    subprocess.run(start_script, shell=True, check=True)
    logging.info("Milvus started successfully.")

# Function to check if Milvus is running and connect
def connect_to_milvus():
    try:
        connections.connect("default", host='localhost', port='19530')
        logging.info("Connected to Milvus successfully.")
    except Exception as e:
        logging.error("Failed to connect to Milvus. Attempting to start Milvus...")
        start_milvus()
        # Wait a moment for Milvus to fully start
        time.sleep(20)
        connections.connect("default", host='localhost', port='19530')
        logging.info("Connected to Milvus after starting the service.")

connect_to_milvus()

def search_similar_stories(query_text, top_k, collection_name, model, output_fields=["post_id", "title", "subreddit", "url"], nprobe=24):
    
    collection = Collection(name=collection_name)

    # Generate the embedding for the query text
    logging.info("Generating query embedding.")
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    
    # Convert NumPy array to list and ensure it's in the format of a list of lists
    query_embedding = query_embedding.tolist()
    if not isinstance(query_embedding[0], list):
        query_embedding = [query_embedding]  # Ensure list of lists for single query

    # Define search parameters
    search_params = {
        "metric_type": "L2",  # or "IP" for inner product
        "params": {"nprobe": nprobe},  # Adjust based on index and search requirements
    }
    
    # Perform the search
    logging.info("Performing the search.")
    results = collection.search(
        data=query_embedding,
        anns_field="embeddings",
        param=search_params,
        limit=top_k,
        output_fields=output_fields  # Fields to retrieve
    )
    
    # # Process and log the search results
    # logging.info("Processing search results.")
    # for hits in results:
    #     for hit in hits:
    #         logging.info(f"Post ID: {hit.entity.get('post_id')}, Title: {hit.entity.get('title')}, "
    #                      f"Subreddit: {hit.entity.get('subreddit')}, URL: {hit.entity.get('url')}, "
    #                      f"Distance: {hit.distance}")
            
    return results
