# python 3.10
#----------------------------------------------------------------------------
# Created By  : Celia Domínguez de Sarriá
# Created Date: 06/05/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Script to create the clasification models, train them and print the accuracy and fscore results""" 
# ---------------------------------------------------------------------------
#IMPORTS

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,KFold
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys
import time
np.set_printoptions(threshold=sys.maxsize)

start=time.time()
#CODE TO TRY ALL CLASSIFFIERS

path = "R:\TFG\INPUT\Test1.csv"

df = pd.read_csv(path,sep=',')

df = df.loc[df["drone_model"] != 1040 ]
df = df.replace(2100,1030) #transform planes to fixed_wing

# df= df.groupby('drone_model')
# df = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True))) #random undersampling

X = df.drop(columns=['drone_model'])
y = df['drone_model']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #, shuffle=True

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

knn =  KNeighborsClassifier(n_neighbors=5) #optimal number of neighbours: 5
knn.fit(X_train, y_train)
y_pred_knn= knn.predict(X_test)

mlp = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(10,10),learning_rate_init=0.02, max_iter = 10000, activation = 'relu', solver = 'adam'))
mlp.fit(X_train,y_train)
y_pred_mlp = mlp.predict(X_test)

rf =  RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred_rf =  rf.predict(X_test)

classifiers = [dt,knn,mlp,rf]
predictions=[y_pred_dt,y_pred_knn,y_pred_mlp,y_pred_rf]
folds=[3,5,10]

for c,p in zip(classifiers,predictions):
  print(type(c).__name__)
  f1 = metrics.f1_score(y_test, p, average='macro')
  f2 = metrics.f1_score(y_test, p, average='micro')
  f3 = metrics.f1_score(y_test, p, average='weighted')
  acc = metrics.accuracy_score(y_test, p)
  print("Accuracy: {:.5}".format(acc))
  print("macro F1-Score: {:.5} \nmicro F1-Score: {:.5} \nweighted F1-Score: {:.5}".format(f1,f2,f3))
  
  # CROSS VALIDATION
  
  # for fold in folds:
  #   score=cross_val_score(c,X_train,y_train,cv=fold)
  #   print("Average Cross Validation score with "+str(fold)+" folds:{:.5}".format(score.mean()))
  
end=time.time()
processingTime=end-start
print ("Processing time: ",processingTime," seconds") 
print('---------------------')

#to save predictions in a txt file
  
# f = open("R:\TFG\INPUT\predicted_"+str(type(svc[1]).__name__)+".txt", "w") 
# print(y_pred_dt, file=f)
# f.close()