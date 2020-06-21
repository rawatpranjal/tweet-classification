# Lexical Features - features from words, without "context".
# Example -> Parts of Speech (POS), Named Entity Recognition (NER)

import pandas as pd
import numpy as np
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')
df = df.sample(df.shape[0], random_state=42)
df.text = df.text_clean.astype(str)
df = df[['id', 'text']]

import spacy
nlp = spacy.load("en_core_web_sm")

df['spacy_doc'] = df['text'].apply(nlp)


def POS_count(x):
    doc = x['spacy_doc']
    N = len([token for token in doc])
    propn_cnt = len([token for token in doc if token.pos_ in ['PROPN']]) / N
    noun_cnt = len([token for token in doc if token.pos_ in ['NOUN']]) / N
    pron_cnt = len([token for token in doc if token.pos_ in ['PRON']]) / N
    num_cnt = len([token for token in doc if token.pos_ in ['NUM']]) / N
    verb_cnt = len([token for token in doc if token.pos_ in ['VERB']]) / N
    symb_cnt = len([token for token in doc if token.pos_ in ['SYM']]) / N
    adp_cnt = len([token for token in doc if token.pos_ in ['ADP']]) / N
    adv_cnt = len([token for token in doc if token.pos_ in ['ADV']]) / N
    punct_cnt = len([token for token in doc if token.pos_ in ['PUNCT']]) / N
    adj_cnt = len([token for token in doc if token.pos_ in ['ADJ']]) / N
    aux_cnt = len([token for token in doc if token.pos_ in ['AUX']]) / N
    conj_cnt = len([token for token in doc if token.pos_ in ['CONJ', 'CCONJ']]) / N
    det_cnt = len([token for token in doc if token.pos_ in ['DET']]) / N

    return propn_cnt, noun_cnt, pron_cnt, num_cnt, verb_cnt, symb_cnt, \
        adp_cnt, adv_cnt, punct_cnt, adj_cnt, aux_cnt, conj_cnt, det_cnt


def NER_count(x):
    doc = x['spacy_doc']
    N = len([token for token in doc])
    gpe_cnt = len([ent for ent in doc.ents if ent.label_ in ['GPE']]) / N
    org_cnt = len([ent for ent in doc.ents if ent.label_ in ['ORG']]) / N
    mon_cnt = len([ent for ent in doc.ents if ent.label_ in ['MONEY']]) / N
    per_cnt = len([ent for ent in doc.ents if ent.label_ in ['PERSON']]) / N
    fac_cnt = len([ent for ent in doc.ents if ent.label_ in ['FAC']]) / N
    loc_cnt = len([ent for ent in doc.ents if ent.label_ in ['LOC']]) / N
    date_cnt = len([ent for ent in doc.ents if ent.label_ in ['DATE']]) / N
    evt_cnt = len([ent for ent in doc.ents if ent.label_ in ['EVENT']]) / N
    prod_cnt = len([ent for ent in doc.ents if ent.label_ in ['PRODUCT']]) / N
    time_cnt = len([ent for ent in doc.ents if ent.label_ in ['TIME', ]]) / N
    perc_cnt = len([ent for ent in doc.ents if ent.label_ in ['PERCENT']]) / N
    quant_cnt = len([ent for ent in doc.ents if ent.label_ in ['QUANTITY', ]]) / N
    ord_cnt = len([ent for ent in doc.ents if ent.label_ in ['ORDINAL', ]]) / N
    card_cnt = len([ent for ent in doc.ents if ent.label_ in ['CARDINAL']]) / N
    norp_cnt = len([ent for ent in doc.ents if ent.label_ in ['NORP']]) / N
    return gpe_cnt, org_cnt, mon_cnt, per_cnt, fac_cnt, loc_cnt, date_cnt, \
        evt_cnt, prod_cnt, time_cnt, perc_cnt, ord_cnt, card_cnt, norp_cnt


df[['propn_cnt', 'noun_cnt', 'pron_cnt', 'num_cnt', 'verb_cnt', 'symb_cnt', 'adp_cnt', 'adv_cnt', 'punct_cnt', 'adj_cnt', 'aux_cnt', 'conj_cnt', 'det_cnt']] = df.apply(POS_count, axis=1, result_type='expand')
df[['gpe_cnt', 'org_cnt', 'mon_cnt', 'per_cnt', 'fac_cnt', 'loc_cnt', 'date_cnt', 'evt_cnt', 'prod_cnt', 'time_cnt', 'perc_cnt', 'ord_cnt', 'card_cnt', 'norp_cnt']] = df.apply(NER_count, axis=1, result_type='expand')

print(df.head())
print(df.columns)
df.drop(['text', 'spacy_doc'], axis=1, inplace=True)
df.to_csv(PATH + 'lexical_features.csv', index=False)
pd.set_option('max_rows', None)
print(df.shape)
print(df.dtypes)
