import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_word_cloud(item_data):
    item_names = " ".join(item['name'] for item in item_data)
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(item_names)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Item Names')
    plt.show()

# Example Usage:
if __name__ == "__main__":
    item_data = [
        {'id': 1, 'name': 'Action Movie'},
        {'id': 2, 'name': 'Comedy Movie'},
        {'id': 3, 'name': 'Romantic Movie'},
        # Add more items here...
    ]
    plot_word_cloud(item_data)