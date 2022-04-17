import numpy as np
import pandas as pd
from naive_bayes import NaiveBayes
from linear_regression import LinearRegression
from adaboost import AdaBoost

"""
pd.options.display.max_columns = 15
pd.set_option('precision', 5)
np.set_printoptions(suppress=True )

data = pd.read_csv('data/drug.csv')
model = NaiveBayes()
model = model.learn(data, class_attribute='Drug',alpha=500)

data_new=pd.read_csv('data/drug_new.csv')
data_new.drop('Unnamed: 6', axis=1,inplace=True)

for i in range(len(data_new)): #lakse da stavis npr range(5) da vidis samo nekoliko predikcija
	prediction, confidence = model.predict(data_new.iloc[i])

	data_new.loc[i,'prediction'] = prediction
	for klasa in confidence:
		data_new.loc[i,'cls_'+klasa] = confidence[klasa]
        #dodajes i kolonu sa confidence-ima
print('------')
print(data_new.iloc[:,-6:])
#print(sum(data_new.prediction==data.Drug)/len(data))
"""
"""
data = pd.read_csv('data/boston.csv')
model=LinearRegression()
model.learn(data,'MEDV', epochs=100, alpha=0.9, lambda_=0, normalize=True)
data_new = pd.read_csv('data/boston.csv')
data_new=data_new.drop('MEDV', axis=1)
print('------')
print('Predictions:')
predictions=model.predict(data_new) 
print(predictions) 

"""

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('data/drugY.csv').iloc[:100,:]
X = data.drop('Drug', axis=1)
y = data['Drug']*2-1
X = pd.get_dummies(X)

model=AdaBoost()
model.learn(X, y, GaussianNB(),
            ensemble_size=10, 
            learning_rate=1,
            base_learner_list=[GaussianNB(), LogisticRegression() ])

model.predict(X, y)
