'''
TEXT CLEAN - cleans up the tweet
'''

# load data
PATH = 'data/'
import pandas as pd
train = pd.read_csv(PATH + 'train.csv', encoding='utf8')
test = pd.read_csv(PATH + 'test.csv', encoding='utf8')
df = train.append(test)
df = df.sample(df.shape[0], random_state=41)

# Clean Text
df['text_clean'] = df['text']
df.text_clean = df.text_clean.str.replace(r'@\w*', 'USERNAME')  # usernames
df.text_clean = df.text_clean.str.replace(r'http\S*', 'WEBSITE')  # usernames
df.text_clean = df.text_clean.str.replace("&amp", '')

df.text_clean = df.text_clean.str.replace(r"[\*\+'\/\(\)\]\[\_\|]", ' ')
df.text_clean = df.text_clean.str.replace(r"[\']", '')
df.text_clean = df.text_clean.str.replace(r"[-&,]", ' ')
df.text_clean = df.text_clean.str.replace(r"[:;?!]", '.')
df.text_clean = df.text_clean.str.replace(r'\.+', '.')
df.text_clean = df.text_clean.str.replace(r'\. \.+', '.')

import unidecode
df.text_clean = df.text_clean.apply(unidecode.unidecode)


def splitter(text):
    import re
    tokens = text.split()
    for i in range(len(tokens)):
        if (tokens[i][0] == '#') or (tokens[i][0] == '@'):
            tokens[i] = tokens[i].replace('#', '')
            tokens[i] = tokens[i].replace('@', '')
            out = re.split(r'(?<=[a-z])(?=[A-Z])', tokens[i])
            tokens[i] = ' '.join(out)
    tokens = ' '.join(tokens)
    return tokens


df.text_clean = df.text_clean.apply(splitter)  # emojis
df.text_clean = df.text_clean.str.replace(r'\s\s+', ' ')  # strip whitespace
df.text_clean.fillna('None', inplace=True)  # fill

'''
TEXT SIMPLE - bare bones, only essential elements.
'''
df['text_simple'] = df['text_clean']

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
stop = stopwords.words('english')


def stopwordRemoval(alist):
    return [i for i in alist if i not in stop]


def rejoin(alist):
    return ' '.join(alist)


df.text_simple = df.text_simple.str.replace(r"USERNAME", '')
df.text_simple = df.text_simple.str.replace(r"WEBSITE", '')
df.text_simple = df.text_simple.str.replace(r".", '')
df.text_simple = df.text_simple.str.replace(r'\b\w{1,2}\b', '')
df['tokens'] = df.text_simple.apply(tknzr.tokenize)  # tokenize
df['tokens'] = df['tokens'].apply(stopwordRemoval)  # remove stopwords
df['text_simple'] = df['tokens'].apply(rejoin)  # untokenize
df.text_simple.fillna('None', inplace=True)  # fill
df.drop('tokens', axis=1, inplace=True)

'''
TEXT LINKS - gets text from the links
import re
import requests
from bs4 import BeautifulSoup
from selectolax.parser import HTMLParser


def getLink(text):
    links = re.findall(r"http\S*", text)
    string = ''
    for link in links:
        try:
            soup = BeautifulSoup(requests.get(link).content, 'html5lib')
            string += soup.title.string
            # string += soup.get_text()
        except:
            pass
    return string


df['text_links'] = df['text'].apply(getLink)
print('\n', df.text_links)
'''

'''
TEXT USERS - get user metadata

import tweepy
import csv
from secrets import TWITTER_C_KEY, TWITTER_C_SECRET, TWITTER_A_KEY, TWITTER_A_SECRET
auth = tweepy.OAuthHandler(TWITTER_C_KEY, TWITTER_C_SECRET)
auth.set_access_token(TWITTER_A_KEY, TWITTER_A_SECRET)
api = tweepy.API(auth)


def get_userinfo(name):
    user = api.get_user(screen_name=name)
    user_info = [name.encode('utf-8'),
                 user.name.encode('utf-8'),
                 user.description.encode('utf-8'),
                 user.followers_count,
                 user.friends_count,
                 user.created_at,
                 user.location.encode('utf-8')]
    return user_info


def getUser(text):
    usernames = re.findall(r'@\w*', text)
    data = []
    for name in usernames:
        try:
            data.append(get_userinfo(name))
        except:
            pass
    return data


df['text_users'] = df['text'].apply(getLink)
'''
print('Cleaned Tweets!')
print('Train:', train.shape, list(train.columns))
print('Test:', test.shape, list(test.columns))
print('Combined:', df.shape, list(df.columns))
df.to_csv(PATH + 'data.csv')
