from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
        ]
tfidf = TFIDF()
X = tfidf.fit_transform(corpus)
print(tfidf.get_feature_names())
print(X.shape)
print(type(X))
print(X.toarray())