import config

import pandas as pd 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score


def run(fold, model):

    df = pd.read_csv(config.TRAINING_FILE_FOLDS)
    
    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    cv_df = df[df['kfold'] == fold].reset_index(drop=True)

    print(cv_df['label'].value_counts())

    ctv = CountVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
    )
    ctv.fit(train_df['tweet'])

    X_train = ctv.transform(train_df['tweet'])
    X_cv = ctv.transform(cv_df['tweet'])

    model = naive_bayes.MultinomialNB()
    model.fit(X_train, train_df['label'])

    preds_cv = model.predict(X_cv)

    print(accuracy_score(cv_df['label'], preds_cv))

if __name__ == '__main__':

    for f in range(0, 1):
        run(fold=f, model='lr')