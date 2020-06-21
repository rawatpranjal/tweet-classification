# NLP-Pipeline-For-Tweet-Classification

A pipeline to solve the problem here: https://www.kaggle.com/c/nlp-getting-started/overview

Best F-Score on leaderboard is near 0.85. 
The best solution here, obtains 0.81 with spacy w2v + GRU/LSTM (with LRELU).
Surprisingly, the bag of words / tweet level features approach with catboost/xgboost gives very similar results: 0.805

Spacy is the library used to generate word vectors.  
Steps:

1. run tweet_clean.py (gives you 2 more "text" columns to operate on).
--> 'text clean' (cleaned tweet; also very good score)
--> 'text simple' (super stripped tweet, essential words only; best score with lstm)
--> 'text links' (text from links in tweet from requests/bs4; works but slow)
--> 'text users' (user meta data like bio/user count, in development)

2. generate tweet level features:

--> keybit_features.py (keybits like 'fire', 'explo', 'burn')
--> word2vec_features.py (avg w2v score for the tweet)
--> lexical_features.py (parts of speech (noun/verbs), named entities (orgs, geopoliticals)
--> sentiment_features.py (objectivity, subjectivity, positivity, negativity)

3. Model training and submission file

--> model_validation.py (cross validation with logistic, xgb)
--> catboost_simple.py (catboost has this amazing processing of categorical and textual features, a personal favorite)
--> pytorch_deepnet.py (deep ANN for classification)
--> pytorch_lstm.py (deep low level LSTM/GRU, that uses spacy

Interpretation/Story Telling: 
< to be done > 


Some findings: 
1. tweet cleaning matters, but dont go overboard. 
2. spacy large embeddings 'core-en-web-lg' is way better than the smaller models
3. Bag of words approach works. As done the embeddings approach. 
4. The best scores on this problem use BERT/Transformers. But its possible to do reasonably well without them. 
5. build a pipeline, then tweak components without affecting complete flow
6. a good cross-validation/evaluation procedure matters. 
