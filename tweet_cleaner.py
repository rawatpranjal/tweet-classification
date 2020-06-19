'''
Input: Train, Test csv
Output: Combined dataset with text_clean column
'''

# Import Packages
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import unidecode
from string import punctuation
punctuation += '´΄’…“”–—―»«'
import warnings
warnings.filterwarnings('ignore')

# Load Data
PATH = 'data/'
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
df = train.append(test)

print(df.isnull().sum())

df = df.sample(df.shape[0], random_state=42,)

# Functions
tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
stop = stopwords.words('english')


def emojiRemover(text):
    return ''.join(c for c in text if c <= '\uFFFF')


def stopwordRemoval(alist):
    return [i for i in alist if i not in stop]


def rejoin(alist):
    return ' '.join(alist)


# Tweet Cleaning
df.text = df.text.str.replace(r'\&\w*;', '')  # html
df.text = df.text.str.replace(r'\$\w*', '')  # cashtags
df.text = df.text.str.replace(r'htmltps?:\/\/.*\/\w*', '')  # links
df.text = df.text.str.replace(r'http?:\/\/.*\/\w*', '')  # links
df.text = df.text.str.replace(r'www?/.*\/\w*', '')  # links
df.text = df.text.str.replace(r'#\w*', '')  # hashtags
df.text = df.text.str.replace(r'[' + punctuation.replace('@', '') + ']+', ' ')
df.text = df.text.str.replace(r'\b\w{1,2}\b', '')  # 2 letter words
df.text = df.text.str.replace(r'\s\s+', ' ')  # strip whitespace
df.text = df.text.str.lstrip(' ')  # strip whitespace
df.text = df.text.apply(emojiRemover)  # emojis
df['tokens'] = df.text.apply(tknzr.tokenize)  # tokenize
df['tokens'] = df['tokens'].apply(stopwordRemoval)  # remove stopwords
df['text'] = df['tokens'].apply(rejoin)  # untokenize
df.text = df.text.str.replace(r'@\S+', '')  # usernames
df.text = df.text.str.replace(r'@', '')  # remove @s
df.text = df.text.str.replace(r'\W +', '')  # only alphanumeric + space
df.text = df.text.apply(unidecode.unidecode)  # remove unicode
df.drop('tokens', axis=1, inplace=True)  # drop tokens
df = train.append(test).merge(df[['id', 'text']], how='left', left_on='id', right_on='id')  # merge
df.columns = ['id', 'keyword', 'location', 'target', 'text', 'text_clean']  # rename
df.text_clean.fillna('None', inplace=True)  # fill
df.text_clean = df.text_clean.astype(str)  # string
df.to_csv(PATH + 'data.csv')  # save

print('Cleaned Tweets!')
print('Train:', train.shape, list(train.columns))
print('Test:', test.shape, list(test.columns))
print('Combined:', df.shape, list(df.columns))
