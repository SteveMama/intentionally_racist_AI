import pickle
import pandas as pd
from dataset import embeddings
from sgd import test_labels


with open('sgd_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

def vecs_to_sentiment(vecs):
    # predict_log_proba gives the log probability for each class
    predictions = model.predict_log_proba(vecs)

    # To see an overall positive vs. negative classification in one number,
    # we take the log probability of positive sentiment minus the log
    # probability of negative sentiment and return it.
    print(predictions[0])
    print(predictions[1])

    # TODO: Write your code here
    return predictions[:, 1] - predictions[:, 0]


def words_to_sentiment(words):
    vecs = embeddings.loc[embeddings.index.isin(words)].dropna()
    log_odds = vecs_to_sentiment(vecs)
    return pd.DataFrame({'sentiment': log_odds}, index=vecs.index)


import re
TOKEN_RE = re.compile(r"\w.*?\b")
# The regex above finds tokens that start with a word-like character (\w), and continues
# matching characters (.+?) until the next word break (\b). It's a relatively simple
# expression that manages to extract something very much like words from text.


def text_to_sentiment(text):
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]

    # Compute sentiments for all the tokens and return the mean value

    # TODO: Write your code here
    print(vecs_to_sentiment(embeddings.loc[tokens]))
    return vecs_to_sentiment(embeddings.loc[tokens]).mean()



print(words_to_sentiment(test_labels).iloc[:20])

print(text_to_sentiment("this example is pretty cool"))


