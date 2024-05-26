import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.generate_stories.similarity_search import search_similar_stories

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# Specify your cache directory path
cache_dir = "/mnt/f/HuggingFace/huggingface_cache"

# Initialize the Sentence Transformer model with a cache directory
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)

def generate_inspired_story(theme, top_k, collection_name, model, openai_model="gpt-3.5-turbo"):
    # Step 1: Retrieve similar stories
    similar_stories = search_similar_stories(theme, top_k, collection_name, model, output_fields=["title", "selftext"])
    
    # Step 2: Extract information and prepare the prompt for GPT-3.5
    story_elements = []
    for hits in similar_stories:
        for hit in hits:
            title = hit.entity.get('title')
            story_text = hit.entity.get('selftext')  # Example of using additional context
            story_elements.append((title, story_text))
    
    prompt = f"""
        STRICT INSTRUCTIONS:

        1. FOLLOW THE INSTRUCTIONS BELOW TO CREATE A SERIOUS, CONTROVERSIAL REDDIT POST INSPIRED BY THE THEMES AFTER YOUR STRICT INSTRUCTIONS.
        
        2. CREATE YOUR OWN STORY, ONLY USE THE TEXT BELOW FOR INSPIRATION.
        
        3. MAKE THE STORY SOUND LIKE A NORMAL PERSON WRITE IT, NOT A PROFESSIONAL WRITER.

        4. MAKE THE STORY SHORT - ONE, OR 2 PARAGRAPH(S) MAXIMUM.

        5. ASK FOR THE AUDIENCES' THOUGHTS ON THIS AT THE END OF YOUR STORY.

        6. FORMAT YOUR POST AS THE FOLLOWING:

        TITLE:

        CONTENT:

        7. PLEASE DO NOT HAVE A RESOLUTION TO THE STORY. LEAVE IT OPEN-ENDED FOR MAXIMUM ENGAGEMENT.

        8. PLEASE ADD MANY DETAILS TO MAKE THE STORY BELIEVABLE.

        {(story_elements)}

        STRICT INSTRUCTIONS REPEATED FOR EMPHASIS:

        1. FOLLOW THE INSTRUCTIONS BELOW TO CREATE A SERIOUS, CONTROVERSIAL REDDIT POST INSPIRED BY THE THEMES AFTER YOUR STRICT INSTRUCTIONS.
        
        2. CREATE YOUR OWN STORY, ONLY USE THE TEXT BELOW FOR INSPIRATION.
        
        3. MAKE THE STORY SOUND LIKE A NORMAL PERSON WRITE IT, NOT A PROFESSIONAL WRITER.

        4. MAKE THE STORY SHORT - ONE, OR 2 PARAGRAPH(S) MAXIMUM.

        5. ASK FOR THE AUDIENCES' THOUGHTS ON THIS AT THE END OF YOUR STORY.

        6. FORMAT YOUR POST AS THE FOLLOWING:

        TITLE:

        CONTENT:

        7. PLEASE DO NOT HAVE A RESOLUTION TO THE STORY. LEAVE IT OPEN-ENDED FOR MAXIMUM ENGAGEMENT.

        8. PLEASE ADD MANY DETAILS TO MAKE THE STORY BELIEVABLE.
        """
    
    # Print length of prompt to ensure it's within the token limit
    logging.info(f"Prompt length: {len(prompt)} characters")

    # Step 3: Generate new story using OpenAI's GPT-3.5
    try:
        response = client.chat.completions.create(model=openai_model,
        messages=[{"role": "system", "content": "You are behind a social media account tasked with creating posts that generate MAXIMUM engagement while keeping the content serious and controversial."},
                  {"role": "user", "content": prompt}],
        max_tokens=1024,  # Adjust based on the desired length of the story
        temperature=0.7,  # Adjust for creativity
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0)
    except Exception as e:
        logging.error(f"Error generating story: {e}")
        return "Error generating story."
    
    return response.choices[0].message.content.strip()

# # Example usage
# query_text = "Relationships"
# top_k = 5
# collection_name = "reddit_stories"
# story = generate_inspired_story(query_text, top_k, collection_name, model)
# print(story)



import pandas as pd

# Initialize a DataFrame to store stories and classifications
df_stories = pd.DataFrame(columns=['Story', 'Classification'])

def manual_classification_and_logging(story):
    # Prompt user for classification
    classification = input("Classify the story as good, bad, or unknown: ").strip().lower()
    while classification not in ['good', 'bad', 'unknown']:
        print("Invalid classification. Please enter 'good', 'bad', or 'unknown'.")
        classification = input("Classify the story as good, bad, or unknown: ").strip().lower()

    # Create a new DataFrame for the current story and classification
    new_entry = pd.DataFrame({'Story': [story], 'Classification': [classification]})

    # Append the new entry to the DataFrame
    global df_stories
    df_stories = pd.concat([df_stories, new_entry], ignore_index=True)


def export_to_csv(append=False):
    filename = "classified_stories.csv"
    
    if append and os.path.exists(filename):
        # Append without including the header if the file already exists
        df_stories.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Write a new file or overwrite the existing file with the header included
        df_stories.to_csv(filename, index=False)
    
    print(f"Data {'appended to' if append else 'exported to'} {filename}.")


# # Modify the example usage part to include classification and logging
# if __name__ == "__main__":
#     query_text = "Relationship Advice"
#     top_k = 5
#     collection_name = "reddit_stories"
#     story = generate_inspired_story(query_text, top_k, collection_name, model)
#     print(story)
    
#     # Manually classify the generated story
#     manual_classification_and_logging(story)
    
#     # Optionally, export the data to CSV
#     # You can call this function whenever you want to save the collected data
#     export_to_csv(append=True)



def generate_multiple_stories(theme, top_k, collection_name, model, n_stories=5):
    stories = []
    for _ in range(n_stories):
        story = generate_inspired_story(theme, top_k, collection_name, model)
        stories.append(story)
    return stories

def evaluate_stories(stories):
    evaluated_stories = []
    for story in stories:
        evaluation_prompt = f"Rate the following story on being interesting, controversial, and normal-sounding: {story}"
        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[{"role": "system", "content": "Evaluate the following story."},
                          {"role": "user", "content": evaluation_prompt}],
                max_tokens=100,
                temperature=0.5
            )
            evaluation = response.choices[0].message.content.strip()
            evaluated_stories.append((story, evaluation))
        except Exception as e:
            logging.error(f"Error evaluating story: {e}")
            evaluated_stories.append((story, "Error evaluating story."))
    return evaluated_stories

def select_best_story(evaluated_stories):
    # Implement your logic to select the best story based on GPT-4 evaluations
    # This is a placeholder function. You might select based on keywords or sentiment analysis
    return evaluated_stories[0][0]  # Placeholder return

if __name__ == "__main__":
    query_text = "Relationship Advice"
    top_k = 5
    collection_name = "reddit_stories"
    n_stories = 3  # Generate 3 stories for evaluation

    stories = generate_multiple_stories(query_text, top_k, collection_name, model, n_stories=n_stories)
    evaluated_stories = evaluate_stories(stories)
    best_story = select_best_story(evaluated_stories)

    print("Best Story:", best_story)