# Import Packages
import pandas as pd
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore')

# Load data
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df['text'] = df['text_clean'].astype(str)
df = df[['id', 'target', 'text']]
df = df.sample(df.shape[0], random_state=42)
with open(PATH + 'keybits.csv', newline='') as f:
    reader = csv.reader(f)
    keybits = list(reader)[0]


def fuzzyratio(x):
    result = []
    for key in keybits:
        if key in x.text:
            result.append([key, x.text.count(key), x.target, x.id])
    return result


def targetScore(x):
    score1, score2, score3 = 0, 0, 0
    cnt = 0.000001
    if len(x.keybit_meta) > 0:
        for i in x.keybit_meta:
            if df_group3[df_group3.keybit == i[0]].shape[0] > 0:
                score1 += df_group3[df_group3.keybit == i[0]].evrate_sample.item()
                score2 += df_group3[df_group3.keybit == i[0]].ocrate_sample.item()
                score3 += df_group3[df_group3.keybit == i[0]].evrate_occurance.item()
                cnt += 1
    return score1 / cnt, score2 / cnt, score3 / cnt


def keybitCount(x):
    cnt = 0
    for i in keybits:
        if i in x:
            cnt += 1
    return cnt


# Extract Keybits, create Tweet-Keybit Database
df['keybit_meta'] = df[['text', 'target', 'id']].apply(lambda x: fuzzyratio(x), axis=1)
keybit_list = df.keybit_meta.tolist()
keybit_list = [item for sublist in keybit_list for item in sublist]
keybit_meta = pd.DataFrame(keybit_list, columns=['keybit', 'count', 'target', 'id'])
keybit_meta = keybit_meta[keybit_meta.target.notna()]  # no leakage

# Keybit level
df_group1 = keybit_meta[['keybit', 'count', 'target']].groupby('keybit', as_index=False).sum()
df_group2 = keybit_meta[['keybit', 'count', 'target']].groupby('keybit', as_index=False).count()
df_group3 = df_group1.merge(df_group2, how='inner', left_on='keybit', right_on='keybit')
df_group3['sampleSize'] = df.shape[0]
df_group3['ocrate_sample'] = df_group3['count_y'] / (df_group3['sampleSize'] + 0.00001)
df_group3['evrate_occurance'] = df_group3['target_y'] / (df_group3['count_y'] + 0.00001)
df_group3['evrate_sample'] = df_group3['target_y'] / (df_group3['sampleSize'] + 0.00001)
df_group3 = df_group3[['keybit', 'ocrate_sample', 'evrate_occurance', 'evrate_sample']]

# Tweet - Keybit Features
df[['orate', 'erate_occur', 'erate_sample']] = df.apply(targetScore, axis=1, result_type='expand')
df['tweet_word_cnt'] = df.text.str.count(' ')
df['tweet_keybit_cnt'] = df.text.apply(keybitCount)
df['tweet_keybit_ratio'] = df['tweet_keybit_cnt'] / (df['tweet_word_cnt'] + 0.00001)


# COUNT VECTORS

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=keybits, analyzer='char_wb', ngram_range=(1, 3))
corpus = df.text.tolist()
X = vectorizer.fit_transform(corpus)
cols = ['word_' + i for i in vectorizer.get_feature_names()]
vecs = pd.DataFrame(X.toarray(), columns=cols)
cols = list(df.columns) + list(vecs.columns)
df = pd.DataFrame(np.hstack((df.values, vecs.values)), columns=cols)
df = df.infer_objects()

# Save
keybit_features = df.drop(['target', 'text', 'keybit_meta'], axis=1)
keybit_features.to_csv(PATH + 'keybit_features.csv', index=False)

pd.set_option('max_rows', None)
print(keybit_features.shape)
print(keybit_features.dtypes)
