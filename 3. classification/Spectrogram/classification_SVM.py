# python 3.10
#----------------------------------------------------------------------------
# Created By  : Celia Domínguez de Sarriá
# Created Date: 06/05/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Script to create the clasification model (only SVM), train it and print the accuracy and fscore results""" 
# ---------------------------------------------------------------------------
#IMPORTS

import pandas as pd
import numpy as np
import time
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
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys

np.set_printoptions(threshold=sys.maxsize)

def get_name(f):
    list_class = [1030.0, 2100.0, 12001.0, 10016.0, 6001.0]
    list_names= ['Fixed_wing','Plane','Octorotor','Quadrotor','Hexarotor']
    return list_names[list_class.index(f)]

path = "R:\TFG\INPUT\V1.csv"

df = pd.read_csv(path,sep=',')

df = df.loc[df["drone_model"] != 1040 ]
df = df.replace(2100,1030) #transform planes to fixed_wing

df= df.groupby('drone_model')
df = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True))) #random undersampling

X = df.drop(columns=['drone_model'])
y = df['drone_model']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

start=time.time()

svc = make_pipeline(StandardScaler(),SVC(decision_function_shape='ovr'))

svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)  

end=time.time()
processingTime=end-start
print ("Processing time: ",processingTime," seconds")
  
f1 = metrics.f1_score(y_test, y_pred_svc, average='macro')
f2 = metrics.f1_score(y_test, y_pred_svc, average='micro')
f3 = metrics.f1_score(y_test, y_pred_svc, average='weighted')
acc = metrics.accuracy_score(y_test, y_pred_svc)
print("Accuracy: {:.5}".format(acc))
print("macro F1-Score: {:.5} \nmicro F1-Score: {:.5} \nweighted F1-Score: {:.5}".format(f1,f2,f3))

#Saving the confusion matrices as images

names = [get_name(i) for i in svc[1].classes_]

plt.figure(figsize=(10,10))
ConfusionMatrixDisplay.from_predictions(y_test,y_pred_svc ,display_labels=names)

image_name='\confusion_matrices'
plt.savefig('R:\TFG\INPUT\PREDICTED\Confusion matrices'+image_name+'_v1.svg')
plt.savefig('R:\TFG\INPUT\PREDICTED\Confusion matrices'+image_name+'_v1.png')
  
# CROSS VALIDATION
  
# for fold in [3,5,10]:
#     score=cross_val_score(svc,X_train,y_train,cv=fold)
#     print("Average Cross Validation score with "+str(fold)+" folds:{:.5}".format(score.mean()))


#to save the predictions in a txt file
  
# f = open("R:\TFG\INPUT\predicted_"+str(type(svc[1]).__name__)+".txt", "w") 
# print(y_pred_svc, file=f)
# f.close()





