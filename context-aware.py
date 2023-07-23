import numpy as np

class ContextAwareRecommender:
    def __init__(self, num_users, num_items, num_features):
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.user_factors = np.random.rand(num_users, num_features)
        self.item_factors = np.random.rand(num_items, num_features)

    def fit(self, user_item_matrix, item_features_matrix, learning_rate=0.01, num_epochs=100):
        for epoch in range(num_epochs):
            for user in range(self.num_users):
                for item in range(self.num_items):
                    if user_item_matrix[user, item] > 0:
                        prediction = np.dot(self.user_factors[user], self.item_factors[item])
                        error = user_item_matrix[user, item] - prediction

                        for feature in range(self.num_features):
                            self.user_factors[user, feature] += learning_rate * (2 * error * self.item_factors[item, feature])
                            self.item_factors[item, feature] += learning_rate * (2 * error * self.user_factors[user, feature])

            for item in range(self.num_items):
                for feature in range(self.num_features):
                    self.item_factors[item, feature] += learning_rate * np.mean(item_features_matrix[item, feature])

    def recommend_items(self, user, num_recommendations=5):
        user_preferences = np.dot(self.user_factors[user], self.item_factors.T)
        sorted_items = np.argsort(user_preferences)[::-1]
        return sorted_items[:num_recommendations]

# Example Usage:
if __name__ == "__main__":
    # Sample user-item matrix (0 indicates no interaction, 1 indicates user-item interaction)
    user_item_matrix = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1],
    ])

    # Sample item features matrix (num_items x num_features)
    item_features_matrix = np.array([
        [0.5, 0.3],
        [0.2, 0.8],
        [0.7, 0.6],
        [0.4, 0.9],
        [0.1, 0.2],
    ])

    num_users, num_items = user_item_matrix.shape
    num_features = item_features_matrix.shape[1]

    recommender = ContextAwareRecommender(num_users, num_items, num_features)
    recommender.fit(user_item_matrix, item_features_matrix)

    user_id = 0
    recommended_items = recommender.recommend_items(user_id, num_recommendations=3)
    print("Recommended Items for User {}: {}".format(user_id, recommended_items))