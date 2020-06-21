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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load data
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df = df[['target', 'id']]

# Load Features
keybit_features = pd.read_csv(PATH + 'keybit_features.csv')
df = df.merge(keybit_features, how='left', left_on='id', right_on='id')

word2vec_features = pd.read_csv(PATH + 'word2vec_features.csv')
df = df.merge(word2vec_features, how='left', left_on='id', right_on='id')

lexical_features = pd.read_csv(PATH + 'lexical_features.csv')
df = df.merge(lexical_features, how='left', left_on='id', right_on='id')

sentiment_features = pd.read_csv(PATH + 'sentiment_features.csv')
df = df.merge(sentiment_features, how='left', left_on='id', right_on='id')

pd.set_option('max_rows', None)
print(df.shape)
print(df.dtypes)

# X-Y Creation
x = df[df.target.notna()].drop(['target', 'id'], axis=1)
x = x.select_dtypes(exclude='object')
y = df['target'][df.target.notna()].astype(int)
print(x.columns)

# Scale & Encode

features = list(x.columns)


def PreProcess(df, cat_features):
    # Features Scaling
    ε = 0.0000000001
    for i in [i for i in features if i not in cat_features]:
        (df[i] - df[i].mean(axis=0) + ε) / (df[i].std(axis=0) + ε)

    # One Hot Encode
    for i in cat_features:
        df = pd.concat([df, pd.get_dummies(df[i], prefix=f'{i}_')], axis=1)
        df.drop(i, axis=1, inplace=True)

    return df


x = PreProcess(x, [])

# Train-Val Separation
import torch as t
t.manual_seed(3)
y = t.tensor(y.values, dtype=t.float)
x = t.tensor(x.values, dtype=t.float)

train_size = 0.8
training_rows = t.LongTensor(round(x.shape[0] * train_size)).random_(0, x.shape[0])
validation_rows = [i for i in list(range(x.shape[0])) if i not in training_rows]
x_train, y_train = x[training_rows], y[training_rows]
x_val, y_val = x[validation_rows], y[validation_rows]

# Network Architechture
n_input, n_output, hidden = x_train.shape[1], 1, x_train.shape[1] + 10
neuralNet = t.nn.Sequential(
    t.nn.Linear(n_input, hidden),
    t.nn.LeakyReLU(),
    t.nn.Linear(hidden, round(hidden / 2)),
    t.nn.LeakyReLU(),
    t.nn.Linear(round(hidden / 2), round(hidden / 3)),
    t.nn.LeakyReLU(),
    t.nn.Linear(round(hidden / 3), round(hidden / 5)),
    t.nn.LeakyReLU(),
    t.nn.Linear(round(hidden / 5), n_output),
    t.nn.Sigmoid()
)

# Optimizer, Weights, Loss Function
optimizer = t.optim.Adam(neuralNet.parameters(), lr=0.005, weight_decay=0.02)
weight = t.tensor(y_train)
weight = t.where(weight >= 0.5, t.tensor(1.0), t.tensor(1.0))
loss_fn = t.nn.BCELoss(weight=weight, reduction='mean')

# Evaluation Metric
from sklearn.metrics import accuracy_score, f1_score


def eval_metric(a, b):
    pred = t.round(a)
    return round(f1_score(pred.detach().numpy(), b.detach().numpy()), 4)


# Train the Network
for i in range(1000):
    loss = loss_fn(neuralNet(x_train), y_train)
    if i % 10 == 0:
        print(i, round(loss.item(), 2), eval_metric(neuralNet(x_train), y_train), eval_metric(neuralNet(x_val), y_val))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Score and Impute
#ypred = neuralNet(t.tensor(x_test.values, dtype=t.float))
#model_test['pred'] = np.where(ypred.detach().numpy() > 0.2, 1, 0)
