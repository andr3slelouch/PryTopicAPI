from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import base64

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, util
from umap import UMAP

import nltk
import re
import pandas as pd
import plotly.io as pio

from keybert import KeyBERT
import os

app = FastAPI()

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedding_model)
embedder = embedding_model  # For consistency in naming

# Define the data models
class FileContent(BaseModel):
    filename: str
    content: str  # Base64 encoded content

class SearchInput(BaseModel):
    search_term: str
    topic_keywords: List[str]  # Topic keywords for relevance calculation
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
    topic_keywords = search_input.topic_keywords  # ["group recommender systems", "recommender systems", "opinion dynamics"]
    files = search_input.files

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Initialize response data
    response = {
        "search_term": search_term,
        # We will add "top_topics", "found_topics", "intertopic_distance_map_html_base64", "hierarchical_clustering_html_base64" below
        "file_results": []  # To store per-file keywords and relevance
    }

    # For topic modeling and visualizations, collect all processed sentences
    all_processed_sentences = []

    # Process each file individually for keywords and relevance
    for file in files:
        try:
            # Decode and read file content
            content_bytes = base64.b64decode(file.content)
            content = content_bytes.decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error decoding file {file.filename}: {str(e)}")

        # Preprocess the text
        preprocessed_text = preprocess_document(content)

        # Split text into sentences
        sentences = nltk.sent_tokenize(preprocessed_text)

        # Remove stopwords from each sentence
        stop_words = set(nltk.corpus.stopwords.words('english'))
        processed_sentences = [
            ' '.join([word for word in sentence.split() if word not in stop_words])
            for sentence in sentences
        ]

        # Add to the overall list for topic modeling
        all_processed_sentences.extend(processed_sentences)

        # Check if processed_sentences is empty
        if not processed_sentences:
            raise HTTPException(status_code=400, detail=f"No valid text data found in file {file.filename} after preprocessing.")

        # Extract keywords from the original content using KeyBERT
        num_keywords = 10  # You can adjust this number
        try:
            keywords = kw_model.extract_keywords(
                content,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=num_keywords,
                use_maxsum=True,
                nr_candidates=20
            )
            extracted_keywords = [kw[0] for kw in keywords]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting keywords for file {file.filename}: {str(e)}")

        # Calculate relevance
        try:
            # Encode keywords
            embeddings_article = embedder.encode(extracted_keywords, convert_to_tensor=True)
            embeddings_topic = embedder.encode(topic_keywords, convert_to_tensor=True)

            # Calculate cosine similarity
            similarities = util.cos_sim(embeddings_article, embeddings_topic)
            max_similarities = similarities.max(dim=1).values
            relevance = max_similarities.mean().item() * 100  # Percentage
            relevance = round(relevance, 2)  # Round to two decimal places
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating relevance for file {file.filename}: {str(e)}")

        # Append per-file results
        file_result = {
            "filename": file.filename,
            "keywords": extracted_keywords,
            "relevance": relevance
        }
        response["file_results"].append(file_result)

    # Check if we have any sentences for topic modeling
    if not all_processed_sentences:
        raise HTTPException(status_code=400, detail="No valid text data found after preprocessing all files.")

    # Topic Modeling with BERTopic on all processed sentences
    try:
        # Adjust UMAP parameters
        umap_model = UMAP(n_neighbors=5, n_components=2, metric='cosine', random_state=42)

        # Create BERTopic instance
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            min_topic_size=2,
            verbose=False
        )

        topic_model.fit_transform(all_processed_sentences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during topic modeling: {str(e)}")

    # Retrieve topic information
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1]  # Exclude outliers

    # Get the top topics (at least three)
    top_n = min(3, len(topic_info))  # Return up to three topics, or fewer if not available
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

    # Generate the intertopic distance map as HTML and encode in base64
    intertopic_fig = topic_model.visualize_topics()
    intertopic_html = intertopic_fig.to_html(full_html=False)
    intertopic_html_base64 = base64.b64encode(intertopic_html.encode('utf-8')).decode('utf-8')

    # Generate the hierarchical clustering as HTML and encode in base64
    hierarchy_fig = topic_model.visualize_hierarchy()
    hierarchy_html = hierarchy_fig.to_html(full_html=False)
    hierarchy_html_base64 = base64.b64encode(hierarchy_html.encode('utf-8')).decode('utf-8')

    # Add the topic modeling results to the response
    response.update({
        "top_topics": top_topics,
        "found_topics": found_topics,
        "intertopic_distance_map_html_base64": intertopic_html_base64,
        "hierarchical_clustering_html_base64": hierarchy_html_base64
    })

    return response
