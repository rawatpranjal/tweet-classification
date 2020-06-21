#  Dependecies
import pandas as pd
import numpy as np

import torch as t
t.manual_seed(42)
t.set_printoptions(precision=4)
if t.cuda.is_available():
    device = t.device('cuda')
else:
    device = t.device('cpu')

import warnings
warnings.filterwarnings('ignore')

'''
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df = df[['target', 'id']]
keybit_features = pd.read_csv(PATH + 'keybit_features.csv')
df = df.merge(keybit_features, how='left', left_on='id', right_on='id')
word2vec_features = pd.read_csv(PATH + 'word2vec_features.csv')
df = df.merge(word2vec_features, how='left', left_on='id', right_on='id')
lexical_features = pd.read_csv(PATH + 'lexical_features.csv')
df = df.merge(lexical_features, how='left', left_on='id', right_on='id')
sentiment_features = pd.read_csv(PATH + 'sentiment_features.csv')
df = df.merge(sentiment_features, how='left', left_on='id', right_on='id')
static = df[df.target.notna()].drop(['target'], axis=1)
static = static.select_dtypes(exclude='object')
pd.set_option('max_rows', None)
print(static.columns, static.shape)
'''

# Load Data
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df.text = df.text_simple.astype(str)
sub = df[df.target.isna()][['id', 'target', 'text']]
df = df[df.target.notna()].fillna('None')
df = df[['id', 'text', 'target']]

import spacy
nlp = spacy.load('en_core_web_lg')
train = df[df.target.notna()].sample(frac=1)

hiddenNeurons = 100
inputNeurons = 300
baseFeatures = 0
outputNeurons = 1
learningRate = 0.001

# GRU Parameters
initHidden = t.zeros(1, hiddenNeurons, requires_grad=True, device=device)
Wf = t.tensor(t.randn(inputNeurons + hiddenNeurons, hiddenNeurons) * 0.01, requires_grad=True, device=device)
Bf = t.zeros(1, hiddenNeurons, requires_grad=True, device=device)
Wr = t.tensor(t.randn(inputNeurons + hiddenNeurons, hiddenNeurons) * 0.01, requires_grad=True, device=device)
Br = t.zeros(1, hiddenNeurons, requires_grad=True, device=device)
Wh = t.tensor(t.randn(inputNeurons + hiddenNeurons, hiddenNeurons) * 0.01, requires_grad=True, device=device)
Bh = t.zeros(1, hiddenNeurons, requires_grad=True, device=device)
Wy = t.tensor(t.randn(inputNeurons + hiddenNeurons + baseFeatures, outputNeurons) * 0.01, requires_grad=True, device=device)
By = t.zeros(1, outputNeurons, requires_grad=True, device=device)
optimizer = t.optim.Adam([Wf, Bf, Wr, Br, Wh, Bh, Wy, By, initHidden], lr=learningRate)
loss_fn = t.nn.BCELoss()

# Load data

from torch import mm, cat
from torch.nn import Sigmoid, LeakyReLU


def iteration(id, doc, target, backprop=True):
    h = [initHidden]
    i = 0
    for token in doc:
        x = t.tensor(token.vector).reshape(1, -1)
        forgotGate = Sigmoid()(mm(cat((x, h[i]), dim=1), Wf) + Bf)
        resetGate = Sigmoid()(mm(cat((x, h[i]), dim=1), Wr) + Br)
        h_candidate = LeakyReLU()(mm(cat((x, resetGate * h[i]), dim=1), Wh) + Bh)
        h_next = forgotGate * h[i] + (1 - forgotGate) * h_candidate
        h.append(h_next)
        i += 1

    #S = t.tensor(static[static.id == id].values, dtype=t.float).reshape(1, -1)
    yprob = Sigmoid()(mm(cat((x, h_next), dim=1), Wy) + By)
    y = t.tensor([target], dtype=t.float)
    loss = loss_fn(yprob, y)
    yhat = t.where(yprob > 0.5, t.ones(1), t.zeros(1))

    if backprop == True:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return round(loss.item(), 2), yhat


train, val = df.iloc[:round(0.8 * df.shape[0])], df.iloc[round(0.8 * df.shape[0]):],
epoch, cnt, trainRight, valRight, trainTotalLoss, valTotalLoss = 0, 0, 0, 0, 0, 0
verbose = 500
while epoch < 20:
    # Train
    id, text, target = train.sample().values[0]
    doc = nlp(text)
    trainLoss, yhat = iteration(id, doc, target)
    trainTotalLoss += trainLoss
    if yhat == target:
        trainRight += 1

    # Validate
    id, text, target = val.sample().values[0]
    doc = nlp(text)
    valLoss, yhat = iteration(id, doc, target, backprop=False)
    valTotalLoss += valLoss
    if yhat == target:
        valRight += 1

    if cnt == verbose:
        epoch += 1
        print(f'Epoch: {epoch}, Train Loss: {trainTotalLoss:0.2f}, Train Acc: {trainRight/verbose:0.2f}, Val Loss: {valTotalLoss:0.2f}, Val Acc: {valRight/verbose:0.2f}')
        cnt, trainRight, valRight, valTotalLoss, trainTotalLoss = 0, 0, 0, 0, 0
    cnt += 1


sub.text = sub.text.fillna('None')
yhats = []
for index, row in sub.iterrows():
    _, yhat = iteration(row.id, nlp(row.text), 1, backprop=False)
    yhats.append(yhat)
sub['target'] = np.array(yhats).astype(int)
sub.drop('text', axis=1, inplace=True)
sub.to_csv(PATH + 'sub_lstm.csv', index=False)
print(sub.shape, sub.columns)
