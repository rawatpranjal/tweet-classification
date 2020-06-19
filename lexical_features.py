# Lexical - without context.
# POS - parts of speech
# Named Entities
# Syntactic dependencies - relation between words


import pandas as pd
import numpy as np
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df = df.sample(df.shape[0], random_state=42)
df.text = df.text_clean.astype(str)
df = df[['id', 'text']]
print(df.isnull().sum())


import spacy
nlp = spacy.load("en_core_web_sm")

df['spacy_doc'] = df['text'].apply(nlp)


def POS_count(x):
    doc = x['spacy_doc']
    propn_cnt = len([token for token in doc if token.pos_ in ['PROPN']])
    noun_cnt = len([token for token in doc if token.pos_ in ['NOUN']])
    pron_cnt = len([token for token in doc if token.pos_ in ['PRON']])
    num_cnt = len([token for token in doc if token.pos_ in ['NUM']])
    verb_cnt = len([token for token in doc if token.pos_ in ['VERB']])
    symb_cnt = len([token for token in doc if token.pos_ in ['SYM']])
    adp_cnt = len([token for token in doc if token.pos_ in ['ADP']])
    adv_cnt = len([token for token in doc if token.pos_ in ['ADV']])
    punct_cnt = len([token for token in doc if token.pos_ in ['PUNCT']])
    return propn_cnt, noun_cnt, pron_cnt, num_cnt, verb_cnt, symb_cnt, adp_cnt, adv_cnt, punct_cnt


def NER_count(x):
    doc = x['spacy_doc']
    gpe_cnt = len([ent for ent in doc.ents if ent.label_ in ['ORG']])
    org_cnt = len([ent for ent in doc.ents if ent.label_ in ['GPE']])
    mon_cnt = len([ent for ent in doc.ents if ent.label_ in ['MONEY']])
    per_cnt = len([ent for ent in doc.ents if ent.label_ in ['PERSON']])
    fac_cnt = len([ent for ent in doc.ents if ent.label_ in ['FAC']])
    loc_cnt = len([ent for ent in doc.ents if ent.label_ in ['LOC']])
    tim_cnt = len([ent for ent in doc.ents if ent.label_ in ['DATE', 'TIME']])
    evt_cnt = len([ent for ent in doc.ents if ent.label_ in ['EVENT']])
    prod_cnt = len([ent for ent in doc.ents if ent.label_ in ['PRODUCT']])
    fact_cnt = len([ent for ent in doc.ents if ent.label_ in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'LOC']])

    return gpe_cnt, org_cnt, mon_cnt, per_cnt, fac_cnt, loc_cnt, tim_cnt, evt_cnt, prod_cnt, fact_cnt


df[['propn_cnt,', 'noun_cnt,', 'pron_cnt,', 'num_cnt,', 'verb_cnt,', 'symb_cnt,', 'adp_cnt,', 'adv_cnt,', 'punct_cnt']] = df.apply(POS_count, axis=1, result_type='expand')
df[['gpe_cnt,', 'org_cnt,', 'mon_cnt,', 'per_cnt,', 'fac_cnt,', 'loc_cnt,', 'tim_cnt,', 'evt_cnt,', 'prod_cnt,', 'fact_cnt']] = df.apply(NER_count, axis=1, result_type='expand')

print(df.head())
print(df.columns)
df.drop('text', axis=1)
df.to_csv(PATH + 'spacy_features.csv')
print(df.shape)
print(df.dtypes)
