# -*- coding: utf-8 -*-
"""Machine Learning Exercise 18 - Natural Language Processing - Emerson Ham
In this exercise we will calculate a variety of feature extraction methods on a news article dataset and use various classifiers to predict the article's category.
We will first use classical methods for feature extraction with Naive Bayes, followed by more recent methods of using word embeddings with a simple Linear SVM model.
"""

# Uncomment the below line to install
# ! pip install spacy
# ! python -m spacy download en_core_web_md

import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.svm import LinearSVC
import numpy as np
import spacy
import en_core_web_md

# Download data from sklearn
data = fetch_20newsgroups(subset="all")
print(data.DESCR)
text = data["data"]
target = data["target"]
print("The following are the 20 topics that an article can belong to:")
print(data["target_names"])

# Split the data into train and test groups
X_train, X_test, y_train, y_test = train_test_split(text, target, random_state=0)
print(f"The training dataset contains {len(X_train)} articles.")
print(f"The test dataset contains {len(X_test)} articles.")

"""Scikit learn implements the BoW feature representation using [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), and it also has implementations for [TF-IDF](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) and [hashed vector](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer) representations.

Determine the feature representations of our dataset using each of those approaches.
"""

# Use English stopwords and produce a BoW representation for the data using up to trigrams
counter = CountVectorizer(stop_words="english", ngram_range = (1,3))
counter.fit(X_train)
X_train_bow = counter.transform(X_train)
X_test_bow = counter.transform(X_test)

# Use the BoW representation to produce a TFIDF representation of the data
tfidfer = TfidfTransformer()
X_train_tfidf = tfidfer.fit_transform(X_train_bow)
X_test_tfidf = tfidfer.transform(X_test_bow)

# Use English stopwords on the OG data and produce a Hashed vector representation for the data using up to trigrams
hasher = HashingVectorizer(stop_words="english", ngram_range = (1,3), alternate_sign=False)
X_train_hash = hasher.fit_transform(X_train)
X_test_hash = hasher.transform(X_test)

# Create naive bayes classifier for each feature
for feat_name, train_feat, test_feat in zip(["Bag of Words", "TF-IDF", "Hashing"],[X_train_bow, X_train_tfidf, X_train_hash], [X_test_bow, X_test_tfidf, X_test_hash]):
    # Create and fit a Multinomial Naive Bayes model
    print(feat_name)
    mnb = MultinomialNB()
    mnb.fit(train_feat, y_train)

    y_pred = mnb.predict(test_feat)
    print(f"Results for {feat_name}")
    print("-"*80)
    print(classification_report(y_test, y_pred))
    print("-"*80)


# Learned Embeddings
# Spacy allows us to parse text and automatically does the following:
# - tokenization
# - lemmatization
# - sentence splitting
# - entity recognition
# - token vector representation

nlp = spacy.load("en_core_web_md")
text = "This is the first sentence in this test string. The quick brown fox jumps over the lazy dog."
parsed_text = nlp(text)

for sent in parsed_text.sents:
    print(f"Analyzing sentence: {sent}")
    print(f"Lemmatization: {sent.lemma_}")
    for token in sent:
        print(f"Analyzing token: {token}")
        if token.is_sent_start:
            print("This token is the first one in the sentence")
        if token.is_stop:
            print("Stop word")
        else:
            print("Not stop word")
        print(f"Entity type: {token.ent_type_}")
        print(f"Part of speech: {token.pos_}")
        print(f"Lemma: {token.lemma_}")
        print("-"*10)
    print("-"*50)

# Tried again with text from my company
my_text = "Inspired by the world's coolest toys, Good Racks makes your gear look its best off the slopes too. What makes us so special? It's all in the details. Good Racks fits right in with the skiing lifestyle. There’s no better way to build hype for the season than to see your gear on display every day."
parsed_text = nlp(my_text)

parsed = nlp(my_text)
for sent in parsed.sents:
    print(f"Analyzing sentence: {sent}")
    print(f"Lemmatization: {sent.lemma_}")
    for token in sent:
        print(f"Analyzing token: {token}")
        if token.is_sent_start:
            print("This token is the first one in the sentence")
        if token.is_stop:
            print("Stop word")
        else:
            print("Not stop word")
        print(f"Entity type: {token.ent_type_}")
        print(f"Part of speech: {token.pos_}")
        print(f"Lemma: {token.lemma_}")
        print("-"*10)
    print("-"*50)

# Glove representation is included with the token, so this is just to show that
print(token.vector)
token.vector.shape

# Glove vectors show how similar words are by putting them close to eachother within that vector space
first_word = "cat" 
second_word = "dog"
print(nlp(first_word).similarity(nlp(second_word)))

# Extract first 1000 articles only since this will take a while to train
new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(X_train[:1000], y_train[:1000], random_state=0)

# Using `nlp` from above, parse every instance of new_X_train
print('\n')
X_train_glove = np.zeros([len(new_X_train), 300])
print('!')
docs = list(nlp.tokenizer.pipe(new_X_train))
n = 0
for item in docs:
    print('.', end='')
    X_train_glove[n][:] = item.vector
    n = n + 1

# repeat for test data
X_test_glove = np.zeros([len(new_X_test), 300])
print('\n!')
docs = list(nlp.tokenizer.pipe(new_X_test))
n = 0
for item in docs:
    print('.', end='')
    X_test_glove[n][:] = item.vector
    n = n + 1
print('')

# Classify sentiment of each article
svm = LinearSVC().fit(X_train_glove, new_y_train)
y_pred = svm.predict(X_test_glove)
print(classification_report(new_y_test, y_pred))