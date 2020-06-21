import pandas as pd
import numpy as np
from textblob import TextBlob
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df = df.sample(df.shape[0], random_state=42)
df.text = df.text_clean.astype(str)
df = df[['id', 'text']]

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
model = SentimentIntensityAnalyzer()


def sentiment_scores(x):
    score = model.polarity_scores(x.text)
    sent_neg, sent_neu, sent_pos, sent_compound = score['neg'], score['neu'], score['pos'], score['compound']
    blob = TextBlob(x.text)
    blob_pol, blob_subj = blob.sentiment.polarity, blob.sentiment.subjectivity
    return sent_neg, sent_neu, sent_pos, sent_compound, blob_pol, blob_subj


df[['sent_neg', 'sent_neu', 'sent_pos', 'sent_compound', 'blob_pol', 'blob_subj']] = df.apply(sentiment_scores, axis=1, result_type='expand')
df.drop('text', axis=1, inplace=True)
df.to_csv(PATH + 'sentiment_features.csv', index=False)
pd.set_option('max_rows', None)
print(df.shape)
print(df.dtypes)
