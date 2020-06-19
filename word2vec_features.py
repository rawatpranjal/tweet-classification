# Load the data
import pandas as pd
import numpy as np
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df = df.sample(df.shape[0], random_state=42)
df.text = df.text_clean.astype(str)

import spacy
nlp = spacy.load("en_core_web_sm")


def word2vec(x):
    doc = nlp(x.text)
    return doc.vector


df.text.fillna('None', inplace=True)
df['avgVec'] = df.apply(word2vec, axis=1)
vecs = df.avgVec.tolist()
vecs = [i.tolist() for i in vecs]
vecs = np.array(vecs)
print(vecs.shape)

cols = ['w2v_' + str(i) for i in range(96)]
cols = ['id'] + cols

word2vec_features = pd.DataFrame(np.c_[df.id.values, vecs], columns=cols)

word2vec_features.to_csv(PATH + 'word2vec_features.csv')
print(df.shape)
print(df.dtypes)
