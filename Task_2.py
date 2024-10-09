# !pip install textblob

from textblob import TextBlob
import pandas as pd

comments_data = {
    'comment': [
        "I love both manga and manhwa! Each has its own charm.",
        "Manhwa art style is so much better than manga.",
        "Manga is overrated. Manhwa stories are more engaging.",
        "I don't like manhwa at all. Manga has way better stories.",
        "Manga and manhwa are just the same to me.",
        "Manhwa has been disappointing lately, the stories are getting repetitive.",
        "Manga art style is unique and unbeatable.",
        "Manga is so boring compared to manhwa!",
        "Both are great but manhwa has really impressed me recently.",
        "I really hate manhwa, it doesn't even compare to manga."
    ]
}

df = pd.DataFrame(comments_data)

def get_sentiment(comment):
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    return 'positive' if polarity > 0 else 'negative'

df['sentiment'] = df['comment'].apply(get_sentiment)

positive_comments = df[df['sentiment'] == 'positive'].shape[0]
negative_comments = df[df['sentiment'] == 'negative'].shape[0]
total_comments = df.shape[0]

positive_percentage = (positive_comments / total_comments) * 100
negative_percentage = (negative_comments / total_comments) * 100

print("Sentiment Analysis Results:")
print(f"Positive comments: {positive_percentage:.2f}%")
print(f"Negative comments: {negative_percentage:.2f}%")

print("\nDetailed Sentiment Analysis:")
print(df)
