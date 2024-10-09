import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'description': [
        "A romantic story between a popular high school girl and a misunderstood boy.",
        "An action-packed adventure where heroes battle monsters to save the world.",
        "In a fantasy realm, a young wizard embarks on a quest to find a magical artifact.",
        "A love triangle unfolds in a high school where friendship and jealousy collide.",
        "A detective investigates mysterious supernatural crimes in the city.",
        "A young woman navigates her career in the fashion industry while dealing with romance.",
        "A group of friends discover they have superpowers and must defend the earth.",
        "A dramatic story of love, betrayal, and revenge in a royal family.",
        "A light-hearted comedy about a quirky group of high school friends and their misadventures.",
        "A fantasy tale of a warrior princess fighting to reclaim her kingdom.",
        "An emotional story of a girl overcoming personal struggles through music.",
        "Two detectives must stop an underground crime syndicate before it's too late.",
        "A magical romance between a fairy and a human, defying the laws of nature.",
        "A high school student discovers he is the reincarnation of a legendary warrior.",
        "A comedy of errors as two roommates accidentally switch lives."
    ],
    'category': [
        'romance', 'action', 'fantasy', 'romance', 'action',
        'romance', 'action', 'drama', 'comedy', 'fantasy',
        'romance', 'action', 'fantasy', 'action', 'comedy'
    ]
}

df = pd.DataFrame(data)


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['description'])
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))


