import config

import re 
import string 
import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score

def remove_stopwords(s):
    stop_words = stopwords.words('english')

    word_list = s.split()
    stopwords_list = list(stop_words)

    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)

    return ' '.join(word_list)

def clean_text(df):
    #s = s.split()
    #s = ' '.join(s)

    df = df.str.lower()
    df = df.apply(lambda x: re.sub('\\n', ' ', str(x)))
    df = df.apply(lambda x: re.sub(r'won\'t', 'will not', str(x)))
    df = df.apply(lambda x: re.sub(r'don\'t', 'do not', str(x)))
    df = df.apply(lambda x: re.sub(r'doesn\'t', 'does not', str(x)))
    df = df.apply(lambda x: re.sub(r'shouldn\'t', 'should not', str(x)))
    df = df.apply(lambda x: re.sub(r'\'s', ' is', str(x)))
    df = df.apply(lambda x: re.sub(r'dont', 'do not', str(x)))
    df = df.apply(lambda x: re.sub(r'\'re', ' are', str(x)))
    df = df.apply(lambda x: re.sub(r'\'t', ' not', str(x)))
    df = df.apply(lambda x: re.sub(r'\'d', 'would', str(x)))
    df = df.apply(lambda x: re.sub(r'\'m', ' am', str(x)))
    df = df.apply(lambda x: re.sub(r'\'ve', ' have', str(x)))
    df = df.apply(lambda x: re.sub(r'\'ll', ' will', str(x)))
    df = df.apply(lambda x: re.sub(r'\`', '\'', str(x)))
    df = df.apply(lambda x: re.sub(r'[0-9]', '', str(x)))
    df = df.apply(lambda x: re.sub(r'\"', '', str(x)))
    df = df.apply(lambda x: re.sub(r'[&]', ' and ', str(x)))
    df = df.apply(lambda x: re.sub(r'[?|!|#|\'|"|@|$|%|^]', r'', str(x)))
    df = df.apply(lambda x: re.sub(r'[.|,|(|)|\[|\]|\{|\}]', r' ', str(x)))
    df = df.apply(remove_stopwords)
    #s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)

    return df

def test_predictions(tf_idf, model):

    test_df = pd.read_csv(config.TEST_FILE)
    #test_df.loc[:, 'tweet'] = test_df['tweet'].apply(clean_text)

    test_df['tweet'] = clean_text(test_df['tweet'])
    test_df.loc[:, 'tweet_length'] = test_df['tweet'].apply(len) 

    X_test = tf_idf.transform(test_df['tweet'])
    #X_test['tweet_length'] = test_df['tweet_length']
    
    preds_test = model.predict(X_test)

    test_submission = pd.DataFrame(
        list(zip(test_df['id'], preds_test)),
        columns = ['id', 'label']
    )

    test_submission.to_csv('../input/test_submission.csv', index=False)

def run(fold, model):

    df = pd.read_csv(config.TRAINING_FILE_FOLDS)
    
    #df.loc[:, 'tweet'] = df['tweet'].apply(clean_text)
    df['tweet'] = clean_text(df['tweet'])
    df.loc[:, 'tweet_length'] = df['tweet'].apply(len) 

    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    cv_df = df[df['kfold'] == fold].reset_index(drop=True)

    """
    ctv = CountVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
    )
    ctv.fit(train_df['tweet'])
    """

    tf_idf = TfidfVectorizer(
        tokenizer=word_tokenize,
        min_df=3, 
        max_features=None, 
        strip_accents='unicode', 
        analyzer='word',
        #token_pattern=r'\w{1,}', 
        ngram_range=(1, 5), 
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1, 
        stop_words = 'english'
    )

    tf_idf.fit(train_df['tweet'])

    X_train = tf_idf.transform(train_df['tweet'])
    X_cv = tf_idf.transform(cv_df['tweet'])
    
    #X_train['tweet_length'] = train_df['tweet_length']
    #X_cv['tweet_length'] = cv_df['tweet_length']

    """
    for c in [1, 5, 10, 50, 100, 1000, 10000]:
        lr = LogisticRegression(C=c).fit(X_train, train_df['label'])
        print ("f1 score for C=%s: %s" % (c, f1_score(cv_df['label'], lr.predict(X_cv))))

    """
    model = LogisticRegression(C=50)
    model.fit(X_train, train_df['label'])
    
    preds_cv = model.predict(X_cv)

    print(f1_score(cv_df['label'], preds_cv))

    test_predictions(tf_idf, model)

if __name__ == '__main__':

    for f in range(0, 1):
        run(fold=f, model='lr')