import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def plot_cosine_similarity_heatmap(item_data):
    item_names = [item['name'] for item in item_data]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(item_names)

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cosine_similarities, annot=True, cmap='coolwarm', xticklabels=item_names, yticklabels=item_names)
    plt.title('Cosine Similarity Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()

# Example Usage:
if __name__ == "__main__":
    item_data = [
        {'id': 1, 'name': 'Action Movie'},
        {'id': 2, 'name': 'Comedy Movie'},
        {'id': 3, 'name': 'Romantic Movie'},
        # Add more items here...
    ]
    plot_cosine_similarity_heatmap(item_data)