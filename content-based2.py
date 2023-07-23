import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
        recommended_items = [self.item_data[i]['name'] for i in similar_items_indices]

        return recommended_items[:num_recommendations]

# Example Usage:
if __name__ == "__main__":
    item_data = [
        {'id': 1, 'name': 'Action Movie'},
        {'id': 2, 'name': 'Comedy Movie'},
        {'id': 3, 'name': 'Romantic Movie'},
        {'id': 4, 'name': 'Adventure Movie'},
        {'id': 5, 'name': 'Sci-Fi Movie'},
        {'id': 6, 'name': 'Horror Movie'},
        {'id': 7, 'name': 'Fantasy Movie'},
        {'id': 8, 'name': 'Animation Movie'},
        {'id': 9, 'name': 'Mystery Movie'},
        {'id': 10, 'name': 'Thriller Movie'},
    ]

    recommender = ContentBasedRecommender(item_data)

    user_preference = "I enjoy action and sci-fi movies"
    recommended_items = recommender.recommend_items(user_preference, num_recommendations=3)
    print("Recommended Items:", recommended_items)
  # Create a word cloud of movie names
    movie_names = ' '.join([item['name'] for item in item_data])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(movie_names)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Popular Movie Genres Word Cloud", fontsize=18)
    plt.show()