class KnowledgeGraphRecommender:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph

    def recommend_items(self, user_profile, num_recommendations=5):
        user_items = set(user_profile)
        candidate_items = set()

        # Traverse the knowledge graph to find related items
        for item in user_items:
            related_items = self.knowledge_graph.get_related_items(item)
            candidate_items.update(related_items)

        # Remove items already present in the user's profile
        candidate_items -= user_items

        # Sort the candidate items based on some relevance score (e.g., popularity)
        sorted_items = sorted(candidate_items, key=lambda item: self.knowledge_graph.get_item_popularity(item), reverse=True)

        return sorted_items[:num_recommendations]

# Example Usage:
if __name__ == "__main__":
    class KnowledgeGraph:
        def __init__(self):
            # Simple representation of a knowledge graph
            self.graph = {
                'item1': {'related_items': ['item2', 'item3'], 'popularity': 10},
                'item2': {'related_items': ['item1', 'item3', 'item4'], 'popularity': 8},
                'item3': {'related_items': ['item1', 'item2', 'item4'], 'popularity': 5},
                'item4': {'related_items': ['item2', 'item3'], 'popularity': 7},
            }

        def get_related_items(self, item):
            return self.graph[item]['related_items']

        def get_item_popularity(self, item):
            return self.graph[item]['popularity']

    knowledge_graph = KnowledgeGraph()
    recommender = KnowledgeGraphRecommender(knowledge_graph)

    user_profile = ['item1', 'item2']
    recommended_items = recommender.recommend_items(user_profile, num_recommendations=3)
    print("Recommended Items:", recommended_items)