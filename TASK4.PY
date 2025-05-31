
# TASK 4: EMAIL SPAM DETECTION WITH MACHINE LEARNING

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample dataset
emails = ["Get cheap meds now!", "Your invoice is attached", "Win a free iPhone", "Meeting at 10 AM"]
labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Input: email text
# Output: 0 (not spam) or 1 (spam)

