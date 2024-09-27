from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import os

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP

import nltk
import re
import pandas as pd
import plotly.io as pio

app = FastAPI()

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Define the data models
class FileContent(BaseModel):
    filename: str
    content: str  # Base64 encoded content

class SearchInput(BaseModel):
    search_term: str
    files: List[FileContent]

# Define the preprocessing function
def preprocess_document(doc):
    # Retain periods for sentence splitting
    doc = re.sub(r'[^a-zA-Z0-9.\s]', '', doc)
    # Convert to lowercase
    doc = doc.lower()
    return doc

@app.post("/search_topic/")
async def search_topic(search_input: SearchInput):
    search_term = search_input.search_term.lower()
    files = search_input.files

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Decode files from base64 and store contents
    texts = []
    for file in files:
        try:
            content_bytes = base64.b64decode(file.content)
            content = content_bytes.decode("utf-8")
            texts.append(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error decoding file {file.filename}: {str(e)}")

    # Preprocess the texts
    docs = []
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for text in texts:
        # Preprocess the entire text
        preprocessed_text = preprocess_document(text)
        # Split text into sentences using NLTK's sentence tokenizer
        sentences = nltk.sent_tokenize(preprocessed_text)
        # Remove stopwords from each sentence
        processed_sentences = [
            ' '.join([word for word in sentence.split() if word not in stop_words])
            for sentence in sentences
        ]
        docs.extend(processed_sentences)

    # Check if docs list is empty
    if not docs:
        raise HTTPException(status_code=400, detail="No valid text data found after preprocessing.")

    # Initialize the embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Adjust UMAP parameters
    umap_model = UMAP(n_neighbors=5, n_components=2, metric='cosine', random_state=42)

    # Create BERTopic instance
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        min_topic_size=2,
        verbose=False
    )

    # Fit the model
    try:
        topic_model.fit_transform(docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during topic modeling: {str(e)}")

    # Retrieve topic information
    topic_info = topic_model.get_topic_info()

    # Exclude outlier topic (-1)
    topic_info = topic_info[topic_info.Topic != -1]

    # Get the top topics (at least three)
    top_n = 3
    top_topics = topic_info.head(top_n).to_dict(orient='records')

    # Find topics containing the search term
    found_topics = []
    for _, row in topic_info.iterrows():
        topic_name = row['Name']
        if search_term in topic_name.lower():
            found_topics.append({
                "Topic_ID": row['Topic'],
                "Topic_Name": row['Name'],
                "Count": row['Count']
            })

    # Generate the intertopic distance map and hierarchical clustering
    intertopic_fig = topic_model.visualize_topics()
    hierarchy_fig = topic_model.visualize_hierarchy()

    # Save figures as PNG images and encode in base64
    try:
        intertopic_fig_bytes = intertopic_fig.to_image(format="png")
        hierarchy_fig_bytes = hierarchy_fig.to_image(format="png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")

    intertopic_base64 = base64.b64encode(intertopic_fig_bytes).decode('utf-8')
    hierarchy_base64 = base64.b64encode(hierarchy_fig_bytes).decode('utf-8')

    # Prepare the response
    response = {
        "search_term": search_term,
        "top_topics": top_topics,
        "found_topics": found_topics,
        "intertopic_distance_map": intertopic_base64,
        "hierarchical_clustering": hierarchy_base64
    }

    return response
