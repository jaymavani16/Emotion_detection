import numpy as np
import pandas as pd

import pickle

from sklearn.ensemble import GradientBoostingClassifier

# fetching the data
train_data = pd.read_csv('./data/features/train_bow.csv')
X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Define and train the Gradient Boost model

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# save
pickle.dump(clf, open('model.pkl','wb'))
