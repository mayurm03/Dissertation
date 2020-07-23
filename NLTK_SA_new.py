from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
#from nltk import FreqDist, classify, NaiveBayesClassifier
import pandas as pd 
from sklearn import preprocessing
import re, string
import nltk
        
def remove_noise(var_tokens, stop_words = ()):
 
    cleaned_tokens = []
 
    for token, tag in pos_tag(var_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
 
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        
        if len(token) > 2 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_sentiment(word,tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

    wn_tag = penn_to_wn(tag)

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [(swn_synset.pos_score()-swn_synset.neg_score())]

def add_sentiment(senti_val):
    return sum(senti_val)

data = pd.read_csv("book-of-kells-1_comments.csv")

pos_val = [None] * 27014
tokens = [None] * 27014
text = data.text
i = 0

for var in text:
    tokens[i] = remove_noise(word_tokenize(var))
    pos_val[i] = nltk.pos_tag(tokens[i])
    i += 1

   
senti_val = [None] * 27014 
i = 0 
del(var)
len_pval = len(pos_val)


while(i<len_pval):
    senti_val[i] = [get_sentiment(x,y) for (x,y) in pos_val[i]]
    i+=1


len_sval = len(senti_val)
i=0


while(i<len_sval):
    senti_val[i] = [add_sentiment(x) for x in senti_val[i]]
    i+=1

df = [sum(a) for a in senti_val]

df = pd.DataFrame(df)
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)

df = pd.DataFrame(df_scaled)

data['Sentiment_score'] = df

data = data.drop(['timestamp','likes','first_reported_at','first_reported_reason','moderation_state','moderated'],axis=1)

data.to_csv(r'G:\Dissertation\Analysis\Run 1 analysis\Sentiment_test1.csv',index = False)
#print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))


