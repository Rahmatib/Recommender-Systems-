import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

class UserBasedCFRecommender:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix

    def calculate_similarity(self):
        # Calculate the cosine similarity between users
        num_users = self.user_item_matrix.shape[0]
        similarity_matrix = np.zeros((num_users, num_users))

        for i in range(num_users):
            for j in range(num_users):
                similarity_matrix[i, j] = 1 - cosine(self.user_item_matrix[i], self.user_item_matrix[j])

        return similarity_matrix

    def recommend_items(self, user_id, num_recommendations=5):
        similarity_matrix = self.calculate_similarity()
        user_vector = self.user_item_matrix[user_id]

        # Calculate weighted average of items based on user similarity
        weighted_scores = np.dot(similarity_matrix[user_id], self.user_item_matrix)
        already_liked_items = set(np.where(user_vector == 1)[0])
        recommended_items = []

        # Get the top N recommended items that the user has not already liked
        for item_id in np.argsort(weighted_scores)[::-1]:
            if item_id not in already_liked_items:
                recommended_items.append(item_id)
                if len(recommended_items) == num_recommendations:
                    break

        return recommended_items

def plot_user_similarity_heatmap(user_item_matrix):
    user_recommender = UserBasedCFRecommender(user_item_matrix)
    similarity_matrix = user_recommender.calculate_similarity()

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title('User Similarity Heatmap')
    plt.xlabel('Users')
    plt.ylabel('Users')
    plt.show()

# Example Usage:
if __name__ == "__main__":
    user_item_matrix = np.array([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1]
    ])

    recommender = UserBasedCFRecommender(user_item_matrix)

    user_id = 0
    recommended_items = recommender.recommend_items(user_id, num_recommendations=3)
    print("Recommended Items for User", user_id, ":", recommended_items)

    # Plot the user similarity heatmap
    plot_user_similarity_heatmap(user_item_matrix)