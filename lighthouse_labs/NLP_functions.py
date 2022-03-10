#Processing smaller functions

def drop_na(dataframe):
    #drop nas
    return dataframe.drop(dataframe[dataframe.isnull().any(axis=1)].index)

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

#Putting it together

def process_features(array, stem=True):
    
    """ Combines all previous pre-processing functions.
    
    INPUT
    -----
    One dimensional array, list, or series containing string sentences
    
    OUTPUT
    ------
    Array with nested arrays containing individual words from original sentence. Removed punctuation, single character words, and converted to lowercase
    """
    
    processed_features = []

    for sentence in array:
        # Remove all the special characters
        in_process = remove_punctuation(sentence)
    
        # Remove all single characters
        for word in in_process:
            if len(word) == 1:
                in_process.remove(word)

        # Converting to Lowercase
        in_process = lowercase(in_process)
        #remove stop words
        in_process = remove_stops(in_process)
         
        #if stemming is set to true then stem text
        if stem is True:
            in_process = stem_text(in_process)
            
        processed_features.append(in_process)
    
    return processed_features

#feature extraction

def similarity_score(dataframe):
    """
    Calculate similarity score for columns within a dataframe
    
    INPUT
    -----
    dataframe with columns 'question1' and 'question2' containing cleaned and tokenized sentences.
    
    OUTPUT
    ------
    array of similarity scores
    """
    
    
    similarity_list = []
    
    #for every row in the dataframe
    for i in range(0, dataframe.shape[0]):
        similarity = 0 #set similarity to 0
        
        #for every word in question 1
        for word in dataframe.iloc[i].question1:
            #print(word)
            
            #if word is also in question 2
            if word in dataframe.iloc[i].question2:
               # print(True)
                similarity += 1
                
        #append row score to list
        similarity_list.append(similarity)
    return similarity_list


def vectors_to_features(dataframe):
    """
    INPUT
    -----
    dataframe with columns 'question1' and 'question2' containing tokenized sentences.
    vectorizor must be fit to larger dataframe before this function.
    
    OUTPUT
    -----
    Sparse matrix with tfidf vectors for question 1, question 2, and a similarity score (words shared)
    """
    
    #transform our question lists seperately
    vectors_q1 = vectorizer.transform(dataframe.question1.values)
    vectors_q2 = vectorizer.transform(dataframe.question2.values)
    
    
    similarity = np.asarray(dataframe.similarity).reshape((dataframe.shape[0],1))
    
    return hstack((vectors_q1, vectors_q2, similarity))

#evaluation

def evaluation(predict, true, save = False):
    """
    INPUT
    -----
    `predict`: array, series or list of predictions from supervised model
    `true`: array, series or list of corresponding true values. Must be same length as predict
    `save`: optional. If specified must be a string specifying name of save file (exclude .png)
    
    OUTPUT
    -----
    List with recall score, precision score, and accuracy score
    Prints confusion matrix with %matplotlib inline
    If `save` is specified saves confusion matrix into folder 'confusion matrix'
    """
    
    CF = ConfusionMatrixDisplay.from_predictions(true, predict).figure_
    
    if save is True:
        CF.savefig(f"confusion_matrices/{save}.png")
        
    return  recall_score(predict, true), precision_score(predict, true), accuracy_score(predict, true)