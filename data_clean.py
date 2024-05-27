import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# 加载数据
data = pd.read_excel('bd_adherence_NOID_V2.xlsx')
print(data.columns)

text_data = data[['Plan', 'Analysis']].fillna('missing')

data['combined_text'] = text_data['Plan'] + ' ' + text_data['Analysis']

train_data = data.loc[:9]
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('spanish')]
    return ' '.join(tokens)


train_data['processed_text'] = train_data['combined_text'].apply(preprocess_text)

# extract feature
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(train_data['processed_text'])


labels = data['Subjective'].fillna(method='ffill')


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


classifier = RandomForestClassifier(n_estimators=10, random_state=42)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
