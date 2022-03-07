def remove_punctuation(text):
    #seperate into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    
    return words


def lowercase(tokenized_text):
    # convert to lower case
    words = [w.lower() for w in tokenized_text]
    return words


def remove_stops(tokenized_text):
    from nltk.corpus import stopwords
    
    stop_words = set(stopwords.words('english'))
    words = [w for w in tokenized_text if not w in stop_words]
    return words

def lemmatize(clean_text):
    from nltk.stem import WordNetLemmatizer
 
    lemmatizer = WordNetLemmatizer()
 
    words = [lemmatizer.lemmatize(w) for w in clean_text]
    return words

def stem_text(clean_text):
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    words = [porter.stem(word) for word in clean_text]
    return words

def preprocess_text(text_array):
    temp = remove_punctuation(text_array)
    temp = lowercase(temp)
    return remove_stops(temp)