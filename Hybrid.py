import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class CollaborativeFilteringRecommender:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix

    def recommend_items(self, user_id, num_recommendations=5):
        user_vector = self.user_item_matrix[user_id]
        predicted_scores = np.dot(self.user_item_matrix, user_vector)
        already_liked_items = set(np.where(self.user_item_matrix[user_id] > 0)[0])
        recommended_items = []

        # Get the top N recommended items that the user has not already liked
        for item_id in np.argsort(predicted_scores)[::-1]:
            if item_id not in already_liked_items:
                recommended_items.append(item_id)
                if len(recommended_items) == num_recommendations:
                    break

        return recommended_items


class ContentBasedRecommender:
    def __init__(self, item_data):
        self.item_data = item_data
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None

    def build_tfidf_matrix(self):
        item_names = [item['name'] for item in self.item_data]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(item_names)

    def recommend_items(self, user_preference, num_recommendations=5):
        if self.tfidf_matrix is None:
            self.build_tfidf_matrix()

        user_preference_vector = self.tfidf_vectorizer.transform([user_preference])
        cosine_similarities = linear_kernel(user_preference_vector, self.tfidf_matrix)

        similar_items_indices = np.argsort(cosine_similarities[0])[::-1]
        recommended_items = [item_id for item_id in similar_items_indices if item_id < len(self.item_data)]

        return recommended_items[:num_recommendations]


class HybridRecommender:
    def __init__(self, user_item_matrix, item_data):
        self.collaborative_filtering_recommender = CollaborativeFilteringRecommender(user_item_matrix)
        self.content_based_recommender = ContentBasedRecommender(item_data)

    def recommend_items(self, user_id, user_preference, num_recommendations=5):
        cf_recommendations = self.collaborative_filtering_recommender.recommend_items(user_id)
        cb_recommendations = self.content_based_recommender.recommend_items(user_preference)

        # Merge recommendations from both recommenders, prioritizing collaborative filtering
        hybrid_recommendations = cf_recommendations + [item_id for item_id in cb_recommendations if item_id not in cf_recommendations]

        return hybrid_recommendations[:num_recommendations]


# Example Usage:
if __name__ == "__main__":
    user_item_matrix = np.array([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1]
    ])

    item_data = [
        {'id': 1, 'name': 'Action Movie'},
        {'id': 2, 'name': 'Comedy Movie'},
        {'id': 3, 'name': 'Romantic Movie'},
        # Add more items here...
    ]

    hybrid_recommender = HybridRecommender(user_item_matrix, item_data)

    user_id = 0
    user_preference = "I enjoy action and adventure movies"
    recommended_items = hybrid_recommender.recommend_items(user_id, user_preference, num_recommendations=3)
    print("Hybrid Recommended Items for User", user_id, ":", recommended_items)
