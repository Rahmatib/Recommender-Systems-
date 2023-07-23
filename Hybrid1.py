import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.spatial.distance import cosine

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

class UserBasedCFRecommender:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix

    def calculate_similarity(self):
        num_users = self.user_item_matrix.shape[0]
        similarity_matrix = np.zeros((num_users, num_users))

        for i in range(num_users):
            for j in range(num_users):
                similarity_matrix[i, j] = 1 - cosine(self.user_item_matrix[i], self.user_item_matrix[j])

        return similarity_matrix

    def recommend_items(self, user_id, num_recommendations=5):
        similarity_matrix = self.calculate_similarity()
        user_vector = self.user_item_matrix[user_id]

        weighted_scores = np.dot(similarity_matrix[user_id], self.user_item_matrix)
        already_liked_items = set(np.where(user_vector == 1)[0])
        recommended_items = []

        for item_id in np.argsort(weighted_scores)[::-1]:
            if item_id not in already_liked_items:
                recommended_items.append(item_id)
                if len(recommended_items) == num_recommendations:
                    break

        return recommended_items

class HybridRecommender:
    def __init__(self, item_data, user_item_matrix):
        self.item_data = item_data
        self.content_recommender = ContentBasedRecommender(item_data)
        self.cf_recommender = UserBasedCFRecommender(user_item_matrix)

    def recommend_items(self, user_id, user_preference, num_recommendations=5, alpha=0.5):
        content_based_scores = self.content_recommender.recommend_items(user_preference, num_recommendations)
        cf_scores = self.cf_recommender.recommend_items(user_id, num_recommendations)

        final_scores = {}
        for item_id in content_based_scores:
            final_scores[item_id] = alpha * (num_recommendations - content_based_scores.index(item_id)) / num_recommendations

        for item_id in cf_scores:
            final_scores[item_id] = final_scores.get(item_id, 0) + (1 - alpha) * (num_recommendations - cf_scores.index(item_id)) / num_recommendations

        recommended_items = sorted(final_scores, key=final_scores.get, reverse=True)

        return recommended_items[:num_recommendations]

# Example Usage:
if __name__ == "__main__":
    item_data = [
        {'id': 1, 'name': 'Action Movie'},
        {'id': 2, 'name': 'Comedy Movie'},
        {'id': 3, 'name': 'Romantic Movie'},
        # Add more items here...
    ]

    user_item_matrix = np.array([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1]
    ])

    hybrid_recommender = HybridRecommender(item_data, user_item_matrix)

    user_id = 0
    user_preference = "I enjoy action movies"
    recommended_items = hybrid_recommender.recommend_items(user_id, user_preference, num_recommendations=3, alpha=0.5)
    print("Hybrid Recommended Items for User", user_id, ":", recommended_items)