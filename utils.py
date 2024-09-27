from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from bertopic import BERTopic

# Load pre-trained TopicBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')


def process_files(texts):
    # Generate embeddings for each text
    embeddings = model.encode(texts)

    # Use KMeans to cluster the embeddings
    num_clusters = len(texts)  # Adjust based on your data
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(embeddings)

    # Get topics and their corresponding texts
    cluster_labels = clustering_model.labels_
    topics = [" ".join([texts[i] for i in range(len(texts)) if cluster_labels[i] == j]) for j in range(num_clusters)]

    # Create a dataframe for visualization
    topics_df = pd.DataFrame({'text': texts, 'cluster': cluster_labels})
    return topics_df, topics


def create_topic_graph(topics_df):
    G = nx.Graph()
    for index, row in topics_df.iterrows():
        G.add_node(row['text'], cluster=row['cluster'])

    # Add edges between nodes with the same cluster
    clusters = topics_df['cluster'].unique()
    for cluster in clusters:
        cluster_texts = topics_df[topics_df['cluster'] == cluster]['text'].tolist()
        for i in range(len(cluster_texts)):
            for j in range(i + 1, len(cluster_texts)):
                G.add_edge(cluster_texts[i], cluster_texts[j])

    return G


def save_plot(graph, filename):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    clusters = [graph.nodes[node]['cluster'] for node in graph.nodes]
    nx.draw(graph, pos, node_color=clusters, with_labels=False, cmap=plt.cm.rainbow, node_size=50, alpha=0.7)
    plt.title("Topic Graph")
    plt.savefig(filename)
    plt.close()
