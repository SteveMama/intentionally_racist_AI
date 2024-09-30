
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset import embeddings,pos_words, neg_words

pos_vectors = embeddings.loc[embeddings.index.isin(pos_words)].dropna()
neg_vectors = embeddings.loc[embeddings.index.isin(neg_words)].dropna()

vectors = pd.concat([pos_vectors, neg_vectors])
targets = np.array([1 for entry in pos_vectors.index] + [-1 for entry in neg_vectors.index])
labels = list(pos_vectors.index) + list(neg_vectors.index)

train_vectors, test_vectors, train_targets, test_targets, train_labels, test_labels = \
    train_test_split(vectors, targets, labels, test_size=0.1, random_state=0)

model = SGDClassifier(loss='log_loss', random_state=0, max_iter=100)
model.fit(train_vectors, train_targets)

with open('sgd_classifier.pkl','wb') as f:
    pickle.dump(model,f)

print(accuracy_score(model.predict(test_vectors), test_targets))
