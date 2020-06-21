# NLP-Pipeline-For-Tweet-Classification

A pipeline to solve the problem here: https://www.kaggle.com/c/nlp-getting-started/overview

* Best F-Score (without cheating) on leaderboard is near 0.85. 
* The best solution here, obtains 0.81 with spacy w2v + GRU/LSTM (with LRELU).
* The bag of words / tweet level features approach with catboost/xgboost gives very similar results: 0.805
* Requirements: Spacy for word vectors, tags; Textblog/nltk for other stuff; Pytorch, sklearn, xgboost, catboost for modelling. 

Steps:

1. run tweet_clean.py (gives you 2 more "text" columns to operate on) and keybits.py (to generate keybits for later use)
* 'text clean' (cleaned tweet; also very good score)
* 'text simple' (super stripped tweet, essential words only; best score with lstm)
* 'text links' (text from links in tweet from requests/bs4; works but slow)
* 'text users' (user meta data like bio/user count, in development)
*  keybits.csv contains keybits within keywords - for the BOW approach

2. generate tweet level features:

* keybit_features.py (keybits like 'fire', 'explo', 'burn')
* word2vec_features.py (avg w2v score for the tweet)
* lexical_features.py (parts of speech (noun/verbs), named entities (orgs, geopoliticals)
* sentiment_features.py (objectivity, subjectivity, positivity, negativity)

3. Model training and submission file

* model_validation.py (cross validation with logistic, xgb)
* catboost_simple.py (catboost has this amazing processing of categorical and textual features, a personal favorite)
* pytorch_deepnet.py (deep ANN for classification)
* pytorch_lstm.py (deep low level LSTM/GRU, that uses spacy

Interpretation/Story Telling: 

1. keywords are essential is distinguishing a general tweet from a disaster related one. Problem is that this challenge is about separating tweets which already have keywords. So almost every single tweet in this data has some keyword that led it to being flagged. 
2. Not all keywords are the same. Some like 'fire' can be used both with 'festival' and 'forest'. But keywords like 'richter' or 'cloudburst' cannot be used in general situations. So just a single feature - target event rate encoding on 'keyword' can give a score of 71F. 
3. Part of Speech/Named entities matters. Disaster related tweets have a lot of nouns, geopolitical entities/organisations. They carry dates and locations, facts and figures. Specifics. While nonDisaster related tweets have a lot of pronouns 'I'/'You'/'Me' and stop words. 
4. While Sentiment-> 'objectivity' or 'subjectivity' score or positivity/negativity for the tweet from textblob should absolutely matter, it was not useful.  
5. Usernames & Links: <TBD>
6. Sequential ordering of words matters. Very sparce words in 'text simple' plus lstm can give as good a score as the bag-of-words approach. 



Some findings: 
1. tweet cleaning matters, but dont go overboard. 
2. spacy large embeddings 'core-en-web-lg' is way better than the smaller models
3. Bag of words approach works. As done the embeddings approach. The former has better interpretation, the latter is faster to impliment. 
4. The best scores on this problem use BERT/Transformers. But its possible to do reasonably well without them. 
5. build a pipeline, then tweak components without affecting complete flow
6. a good cross-validation/evaluation procedure matters. 
