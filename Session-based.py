import numpy as np

class SessionBasedRecommender:
    def __init__(self, session_data):
        self.session_data = session_data
        self.item_similarity = None

    def build_item_similarity(self):
        num_items = max(item for session in self.session_data for item in session) + 1
        item_count = np.zeros((num_items, num_items))

        for session in self.session_data:
            for i in range(len(session) - 1):
                item_count[session[i], session[i+1]] += 1

        norm = np.sqrt(np.sum(item_count, axis=1))
        epsilon = 1e-8  # Small epsilon value to avoid division by zero
        self.item_similarity = item_count / ((norm[:, None] + epsilon) * (norm[None, :] + epsilon))

    def recommend_items(self, current_session, num_recommendations=5):
        if self.item_similarity is None:
            self.build_item_similarity()

        item_scores = np.zeros(len(self.item_similarity))
        for item in current_session:
            item_scores += self.item_similarity[item]

        top_items = np.argsort(item_scores)[::-1]
        recommended_items = [item for item in top_items if item not in current_session]

        return recommended_items[:num_recommendations]

# Example Usage:
if __name__ == "__main__":
    session_data = [
        [1, 2, 3],
        [4, 2, 5, 6],
        [7, 8],
        [2, 9, 10, 11]
    ]

    recommender = SessionBasedRecommender(session_data)

    current_session = [2, 3]
    recommended_items = recommender.recommend_items(current_session, num_recommendations=3)
    print("Recommended Items:", recommended_items)