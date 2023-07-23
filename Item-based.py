import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCollaborativeFiltering:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.user_item_matrix = np.zeros((num_users, num_items))

    def add_interaction(self, user, item, rating):
        self.user_item_matrix[user, item] = rating

    def build_item_similarity(self):
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)

    def recommend_items(self, user, num_recommendations=5):
        if not hasattr(self, 'item_similarity'):
            self.build_item_similarity()

        user_preferences = self.user_item_matrix[user]
        item_scores = np.dot(user_preferences, self.item_similarity)

        # Remove items already interacted by the user
        item_scores[self.user_item_matrix[user] > 0] = -np.inf

        recommended_items = np.argsort(item_scores)[::-1]
        return recommended_items[:num_recommendations]

# Example Usage:
if __name__ == "__main__":
    num_users = 4
    num_items = 5

    recommender = ItemBasedCollaborativeFiltering(num_users, num_items)

    # Sample user-item interactions
    recommender.add_interaction(0, 0, 5)
    recommender.add_interaction(0, 1, 3)
    recommender.add_interaction(0, 2, 4)
    recommender.add_interaction(1, 0, 1)
    recommender.add_interaction(1, 2, 2)
    recommender.add_interaction(2, 1, 3)
    recommender.add_interaction(2, 3, 4)
    recommender.add_interaction(3, 0, 2)
    recommender.add_interaction(3, 1, 5)
    recommender.add_interaction(3, 2, 1)
    recommender.add_interaction(3, 3, 3)
    recommender.add_interaction(3, 4, 4)

    user_id = 0
    recommended_items = recommender.recommend_items(user_id, num_recommendations=3)
    print("Recommended Items for User {}: {}".format(user_id, recommended_items))