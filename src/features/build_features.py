import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from /data/processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

# filling the missing values, It is generally a part of data preprocessing but we are practicing pipeline building so i am doing it here.
train_data.fillna('',inplace=True)
test_data.fillna('',inplace=True)

# splitting X_train, y_train, X_test and y_test
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

data_path = os.path.join('data','features')
os.makedirs(data_path)
train_df.to_csv(os.path.join(data_path,'train_bow.csv'))
test_df.to_csv(os.path.join(data_path,'test_bow.csv'))