#importing the modules required

import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report 
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge

#loading the file

df=pd.read_csv('train mobile.csv')
len(df)

#spliting data into train-valid-test sets

train,valid,test=np.split(df.sample(frac=1),[int((0.7)*len(df)),int((0.85)*len(df))])
b=df.price_range.unique()
cols=df.columns

#scaling function
def scale_dataset(data,oversample=False):
    x=data[data.columns[:-1]].values
    y=data[data.columns[-1]].values
    scalar=StandardScaler()
    x=scalar.fit_transform(x)
    if oversample:
        ros=RandomOverSampler()
        x,y=ros.fit_resample(x,y)
    data=np.hstack((x,np.reshape(y,(-1,1))))
    return data,x,y

#scaling the data
train,xtrain,ytrain=scale_dataset(train,oversample=False)
valid,xvalid,yvalid=scale_dataset(valid,oversample=False)
test,xtest,ytest=scale_dataset(test,oversample=False)


#creating linear regression model
Bayes_model=BayesianRidge()

#training the linear model
Bayes_model=Bayes_model.fit(xtrain,ytrain)


y_pred=Bayes_model.predict(xvalid)

score=r2_score(yvalid,y_pred)

ytestpred=Bayes_model.predict(xtest)

test_score=r2_score(ytest,ytestpred)

#pickling data into the pickle
file='pickltest.pkl'
file_obj=open(file,'wb')
pkl.dump(Bayes_model,file_obj)
file_obj.close()


ns=pkl.load(fileobj)

