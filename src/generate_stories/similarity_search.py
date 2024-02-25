import logging
import pexpect
import subprocess
import time
import sys

from pymilvus import (
    connections, Collection,
)

# Function to start Milvus
# def start_milvus(start_script="./vector_database/standalone_embed.sh start"):
#     # Start Milvus
#     subprocess.run(start_script, shell=True, check=True)
#     logging.info("Milvus started successfully.")

def start_milvus(start_script="./vector_database/standalone_embed.sh start"):
    try:
        # Start Milvus with potential need for password
        child = pexpect.spawn(start_script, encoding='utf-8')
        child.logfile = sys.stdout  # Optionally log output to stdout

        # Wait for potential password prompt
        i = child.expect(['Password:', pexpect.EOF, pexpect.TIMEOUT], timeout=30)

        if i == 0:  # 'Password:' found
            password = '41288202256'  # Replace with your actual password or securely retrieve it
            child.sendline(password)
            child.expect(pexpect.EOF)  # Wait for the script to finish

        elif i == 1:  # EOF
            logging.info("Milvus started successfully without needing a password.")

        elif i == 2:  # Timeout
            logging.error("Timeout occurred while starting Milvus.")

    except pexpect.exceptions.ExceptionPexpect as e:
        logging.error(f"Failed to start Milvus: {e}")

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

connect_to_milvus()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # Process and log the search results
    logging.info("Processing search results.")
    for hits in results:
        for hit in hits:
            logging.info(f"Post ID: {hit.entity.get('post_id')}, Title: {hit.entity.get('title')}, "
                         f"Subreddit: {hit.entity.get('subreddit')}, URL: {hit.entity.get('url')}, "
                         f"Distance: {hit.distance}")
            
    return results
