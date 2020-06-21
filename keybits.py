# Load the data
import pandas as pd
PATH = 'data/'
df = pd.read_csv(PATH + 'data.csv')

keywords = list(set(df[(df.keyword.notna()) & (df.target.notna())].keyword))
keywords = keywords + ['tsunami', 'natural disasters', 'volcano', 'tornado', 'avalanche', 'earthquake', 'blizzard', 'drought', 'bushfire', 'tremor', 'dust storm', 'magma', 'twister', 'windstorm', 'heat wave', 'cyclone', 'forest fire', 'flood', 'fire', 'hailstorm', 'lava', 'lightning', 'high pressure', 'hail', 'hurricane', 'seismic', 'erosion', 'whirlpool', 'Richter scale', 'whirlwind', 'cloud', 'thunderstorm', 'barometer', 'gale', 'blackout', 'gust', 'force', 'low-pressure', 'volt', 'snowstorm', 'rainstorm', 'storm', 'nimbus', 'violent storm', 'sandstorm', 'casualty', 'Beaufort scale', 'fatal', 'fatality', 'cumulonimbus', 'death', 'lost', 'destruction', 'money', 'tension', 'cataclysm', 'damage', 'uproot', 'underground', 'destroy', 'arsonist', 'wind scale', 'arson', 'rescue', 'permafrost', 'disaster', 'fault', 'scientist', 'shelter']

keywords = [i.replace(r'%20', '') for i in keywords]


'''
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = df[df.target == 1].text.str.replace(r'\W +', '').to_string().split('\n')
vectorizer = TfidfVectorizer(max_features=1000,
                             ngram_range=(1, 2),
                             stop_words="english")
X = vectorizer.fit_transform(corpus)
top_words = vectorizer.get_feature_names()
# print(top_words)
print(vectorizer.vocabulary_)
temp = pd.DataFrame.from_dict(vectorizer.vocabulary_, orient='index')
temp = temp.reset_index()
temp.columns = ['keyword', 'count']
temp = temp.sort_values(by='count', ascending=False)
print(temp.to_string(index=False))
'''

# manually draw words
newWords = ['zionist', 'youtube', 'atomic', 'yazidis', 'wreck', 'wound', 'suspect', 'worst', 'wildfire', 'whirlwind', 'weather', 'weapon', 'wave', 'rescue', 'warning', 'war', 'volcano', 'violent', 'victim', 'terror', 'devastate', 'strike', 'trauma', 'trap', 'bioterror', 'train derailment', 'tragedy', 'traffic', 'tornado', 'today', 'tomorrow', 'yesterday', 'threaten', 'thunderstorm', 'thunder', 'terrorist', 'militant', 'tension', 'killed', 'dead', 'survive', 'suicide bomb', 'suicide', 'struggle', 'strong', 'storm', 'spill', 'soldier', 'smoke', 'site', 'sink', 'shoot', 'forest', 'severe', 'seismic', 'security', 'scene', 'save', 'rock', 'road', 'riot', 'responder', 'resident', 'rescuer', 'med', 'reports', 'oil', 'refugee', 'horror', 'recount', 'record', 'reactor', 'raze', 'quarantine', 'prosecute', 'property', 'damage', 'plane', 'debris', 'crash', 'bodies', 'died', 'passenger', 'pandemonium', 'palestinian', 'pakistan', 'outrage', 'officer', 'occurred', 'declare', 'nuclear', 'notice', 'stab', 'natural disaster', 'muslims', 'murder', 'emergency', 'mudslide', 'missing', 'metro', 'escape', 'massacre', 'mass murder', 'manslaughter', 'lightning', 'catastrophic', 'disease', 'landslide', 'hazard', 'issues', 'injure', 'investigator', 'islamic', 'incident', 'fear', 'hostage', 'horrible', 'home', 'heat', 'hate', 'government', 'gaza', 'force', 'flood', 'flash', 'flight', 'firefighter', 'drown', 'family', 'fatalities', 'eyewitness', 'fact', 'explosion', 'examine', 'evacuation', 'engulf', 'electrocute', 'effect', 'dust', 'drought', 'dozens', 'hundreds', 'disrupt', 'displace', 'disaster', 'detonate', 'destroy', 'demolition', 'crisis', 'cyclone', 'criminal', 'country', 'cost', 'conclusive', 'confirm', 'collision', 'collapse', 'climate', 'coast', 'claims', 'chemical', 'cause', 'casualty', 'bus', 'burn', 'bridge', 'attack', 'arson', 'ashes', 'arrest', 'area', 'army', 'annihilate', 'amid', 'aircraft', 'air', 'year', 'km']


keywords = keywords + newWords
keywords = list(set(keywords))
# Clean Keywords
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
lemmatizer = WordNetLemmatizer()

to_remove = ['annihilated', 'annihilation', 'airplaneaccident', 'arsonist', 'bloody', 'bodybags', 'bombed', 'bombing', 'bodybagging', 'bridgecollapse', 'buildingsburning', 'buildingsonfire', 'burned', 'burn', 'burningbuildings','chemicalemergency', 'clifffall', 'collapsed', 'collided', 'collision', 'crashed','derailed', 'derailment', 'desolation','destroyed','detonation',  'devastated', 'devastation', 'displaced','drowned', 'drowning','dust storm', 'duststorm','electrocuted','emergencyplan', 'emergencyservices', 'engulfed',  'evacuated', 'evacuation','exploded', 'explosion', 'fatality', 'flattened', 'forest fire', 'forestfires', 'hailstorm', 'heat wave', 'hellfire', 'hijacker', 'hijacking',  'injured', 'injury', 'inundation','natural disaster', 'natural disasters', 'nucleardisaster', 'obliterated', 'obliteration', 'oil', 'quarantined','razed', 'rescued', 'rescuer','rioting','screamed', 'screaming','sinking',  'suicide bomb', 'suicidebomb', 'suicidebombing', 'survived', 'survivor',  'terrorism', 'terrorist','threaten', 'thunderstorm','train derailment', 'traumatised',  'violent storm', 'violentstorm',  'wounded','wreckage', 'wrecked', 'bushfires', 'catastrophic', 'rainstorm', 'bus', 'radiationemergency', 'naturaldisaster', 'forestfire', 'trainderailment', 'structuralfailure', 'nuclearreactor', 'mass murder', 'Richter scale']

to_add = ['airplane', 'bushfire', 'service', 'flatten', 'hell' ,'thousands', 'radiation', 'pummel', 'trainderailment', 'gunman', 'loud', 'bang', 'blow', 'blast', 'boom', 'rain', 'wind', 'scale', 'beaufort', 'richter', 'car', 'train', 'mass', 'cycle']

# Apply Stemmer
ps = PorterStemmer()
keywords = [i for i in keywords + to_add if i not in to_remove]
keywords = list(set([ps.stem(i.lower()) for i in keywords]))
keywords.sort()
print('No. of keybits: ', len(keywords))

# Save keybits
import csv
with open(PATH+'keybits.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(keywords)

