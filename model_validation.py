# Import Packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df = df[['target', 'id']]

# Load Features
keybit_features = pd.read_csv(PATH + 'keybit_features.csv')
print(keybit_features.shape)
df = df.merge(keybit_features, how='left', left_on='id', right_on='id')

word2vec_features = pd.read_csv(PATH + 'word2vec_features.csv')
df = df.merge(word2vec_features, how='left', left_on='id', right_on='id')

spacy_features = pd.read_csv(PATH + 'spacy_features.csv')
df = df.merge(spacy_features, how='left', left_on='id', right_on='id')

# X-Y Creation
print(df.columns)
x = df[df.target.notna()].drop(['target', 'id'], axis=1)
x = x.select_dtypes(exclude='object')
y = df['target'][df.target.notna()]
print(x.columns)

# Cross Validation


def CrossValidation(model):
    S = 10
    sss = StratifiedShuffleSplit(n_splits=S, test_size=0.3)
    sss.get_n_splits(x, y)
    score = 0
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(x_train.shape, x_test.shape)
        model.fit(x_train, y_train)
        score += f1_score(y_test, model.predict(x_test)) / S
    print(score)


model = LogisticRegression(C=5)
CrossValidation(model)

#model = MLPClassifier(learning_rate_init=0.01, learning_rate='adaptive', max_iter=1000)
# CrossValsidation(model)

model = XGBClassifier(max_depth=8, reg_lambda=5)
CrossValidation(model)

print(df.columns)
# Submission
x = df[df.target.isna()].drop(['target', 'id'], axis=1)
x = x.select_dtypes(exclude='object')

sub = df[df.target.isna()][['id', 'target']]
sub['target'] = model.predict(x).astype(int)
sub.to_csv(PATH + 'sub.csv', index=False)
print(sub.shape, sub.columns)
