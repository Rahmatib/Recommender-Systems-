import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, item_data):
        self.item_data = item_data
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None  # Add this line

    def build_tfidf_matrix(self):
        item_names = [item['name'] for item in self.item_data]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # Modify this line
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(item_names)

    def recommend_items(self, user_preference, num_recommendations=5):
        if self.tfidf_matrix is None:
            self.build_tfidf_matrix()

        # Use the same vectorizer for user_preference
        user_preference_vector = self.tfidf_vectorizer.transform([user_preference])  # Modify this line
        cosine_similarities = linear_kernel(user_preference_vector, self.tfidf_matrix)

        similar_items_indices = np.argsort(cosine_similarities[0])[::-1]
        recommended_items = [self.item_data[i]['name'] for i in similar_items_indices]

        return recommended_items[:num_recommendations]

# Example Usage:
if __name__ == "__main__":
    item_data = [
        {'id': 1, 'name': 'Action Movie'},
        {'id': 2, 'name': 'Comedy Movie'},
        {'id': 3, 'name': 'Romantic Movie'},
        # Add more items here...
    ]

    recommender = ContentBasedRecommender(item_data)

    user_preference = "I enjoy action movies"
    recommended_items = recommender.recommend_items(user_preference, num_recommendations=1)
    print("Recommended Items:", recommended_items)