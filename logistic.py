import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

social=pd.read_csv('/Users/bunny/Desktop/dataset/SocialNetworkAds.csv')

X=social.iloc[:,2].values
y=social.iloc[:,3].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

print(y_test)

# from sklearn.metrics import confusion_matrix
# cn=confusion_matrix()
# cn(y_test,y_pred)
# print(cn)
