import numpy as np

class SocialCFRecommender:
    def __init__(self, user_item_matrix, social_network_matrix):
        self.user_item_matrix = user_item_matrix
        self.social_network_matrix = social_network_matrix

    def calculate_similarity(self):
        # Calculate the cosine similarity between users based on social connections
        num_users = self.social_network_matrix.shape[0]
        similarity_matrix = np.zeros((num_users, num_users))

        for i in range(num_users):
            for j in range(num_users):
                similarity_matrix[i, j] = np.dot(self.social_network_matrix[i], self.social_network_matrix[j]) / \
                                         (np.linalg.norm(self.social_network_matrix[i]) *
                                          np.linalg.norm(self.social_network_matrix[j]))

        return similarity_matrix

    def recommend_items(self, user_id, num_recommendations=5):
        similarity_matrix = self.calculate_similarity()
        user_vector = self.user_item_matrix[user_id]
        social_similarities = similarity_matrix[user_id]

        # Calculate weighted average of items based on user similarity and social similarities
        weighted_scores = np.dot(social_similarities, self.user_item_matrix)
        already_liked_items = set(np.where(user_vector == 1)[0])
        recommended_items = []

        # Get the top N recommended items that the user has not already liked
        for item_id in np.argsort(weighted_scores)[::-1]:
            if item_id not in already_liked_items:
                recommended_items.append(item_id)
                if len(recommended_items) == num_recommendations:
                    break

        return recommended_items


# Example Usage:
if __name__ == "__main__":
    user_item_matrix = np.array([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1]
    ])

    social_network_matrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 1]
    ])

    recommender = SocialCFRecommender(user_item_matrix, social_network_matrix)

    user_id = 0
    recommended_items = recommender.recommend_items(user_id, num_recommendations=3)
    print("Recommended Items for User", user_id, ":", recommended_items)