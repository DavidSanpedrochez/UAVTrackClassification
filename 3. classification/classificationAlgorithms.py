################################
# This file includes the classification funcionalities of the UAV Track Classification project.
# 
# @author: David Sanchez <davsanch@inf.uc3m.es>
# @author: Daniel Amigo <damigo@inf.uc3m.es>
################################

#############################################################
#                       IMPORTS                             #
#############################################################
import math
import os
import pandas as pd
from cmath import nan
import matplotlib.pyplot as mp
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump, load
from os.path import isfile, join                            # To use the data files
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
##### Algorithms imports #####
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


#############################################################
#                      ALGORITHMS                           #
#############################################################
# *****************************************************************************************
# ** Decision tree Algorithm
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT:
# ******* OUTPUT: 
# *****************************************************************************************

# INTENTO DE PINTAR MATRIZ DE CONFUSION BUENA (CON PORCENTAJES)
# create a new model 
def decisionTree(maxDepth,ExperimentDir,X_train, y_train, X_test, y_test, outputPath,imagesDir,modelsDir,resultsDir, printResults, showModel, saveResults, saveModel):
      dt = DecisionTreeClassifier(random_state=0,max_depth=maxDepth)
      dt.fit(X_train, y_train)
      y_pred = dt.predict(X_test)
      y_pred_proba = dt.predict_proba(X_test)#[:, 1]

      #ANALISIS DE RESULTADOS
      confmat = metrics.confusion_matrix(y_test, y_pred)
      acc = metrics.accuracy_score(y_test, y_pred)
      precision = metrics.precision_score(y_test, y_pred, average='micro')
      recall = metrics.recall_score(y_test, y_pred, average='micro')
      f1 = metrics.f1_score(y_test, y_pred, average='micro')
      #auroc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
      if printResults:      
            print(f"Accuracy score: {acc:.5f}\n" + f"Recall: {recall:.5f}\n" + f"F1 score: {f1:.5f}\n")
            fig = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=dt.classes_)
            fig.plot()
            # save figure to PNG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"PNG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "decisionTreeConfusionMatrix_"+ExperimentDir+"_depth_"+str(maxDepth)+".png"
            fig.figure_.savefig(join(DIR,FILE))
            # save figure to SVG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"SVG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "decisionTreeConfusionMatrix_"+ExperimentDir+"_depth_"+str(maxDepth)+".svg"
            fig.figure_.savefig(join(DIR,FILE))
            
      if showModel:
            # INTENTO DE PINTAR EL ÁRBOL DE DECISIÓN
            fig = plt.figure(figsize=(30,10)) #Para cambiar el tamaño del árbol
            tree.plot_tree(dt, max_depth=3, feature_names=X_train.columns, fontsize= 11, filled=True)
            # save figure to PNG file
            DIR= join(outputPath,ExperimentDir,imagesDir,"PNG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "decisionTreeGeneratedTree_"+ExperimentDir+"_depth_"+str(maxDepth)+".png"
            fig.savefig(join(DIR,FILE))
            # save figure to SVG file
            DIR= join(outputPath,ExperimentDir,imagesDir,"SVG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "decisionTreeGeneratedTree_"+ExperimentDir+"_depth_"+str(maxDepth)+".svg"
            fig.savefig(join(DIR,FILE))
            plt.close(fig)
      if saveResults:
            #save results
            #results = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
            DIR=join(outputPath,ExperimentDir,resultsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            #FILE="decisionTreeResults_"+ExperimentDir+"_depth_"+str(maxDepth)+".csv"
            #results.to_csv(join(DIR,FILE))
            # create txt with accuracy and recall
            FILE="decisionTreeResults_"+ExperimentDir+"_depth_"+str(maxDepth)+".txt"
            with open(join(DIR,FILE), "w") as text_file:
                  text_file.write(f"Confusion matrix:\n{confmat}\n" +
                  f"Accuracy score: {acc:.5f}\n" +
                  f"Precision: {precision:.5f}\n"+
                  f"Recall: {recall:.5f}\n" +
                  f"F1 score: {f1:.5f}\n")
      if saveModel:
            # save model
            DIR=join(outputPath,ExperimentDir,modelsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE="decisionTreeModel_"+ExperimentDir+"_depth_"+str(maxDepth)+".joblib"
            dump(dt, join(DIR,FILE))
      return confmat,acc,precision,recall,f1

# TO COMPLETE


      
# *****************************************************************************************
# ** Random forest Algorithm
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT:
# ******* OUTPUT: 
# *****************************************************************************************
# create a new model
def randomForest(maxDepth,nEstimators,ExperimentDir,X_train, y_train, X_test, y_test, outputPath,imagesDir,modelsDir,resultsDir, printResults, showModel, saveResults, saveModel):
      rf = RandomForestClassifier(random_state=0,max_depth=maxDepth,n_estimators=nEstimators)
      rf.fit(X_train, y_train)
      y_pred = rf.predict(X_test)
      y_pred_proba = rf.predict_proba(X_test)#[:, 1]
      #ANALISIS DE RESULTADOS
      confmat = metrics.confusion_matrix(y_test, y_pred)
      acc = metrics.accuracy_score(y_test, y_pred)
      precision = metrics.precision_score(y_test, y_pred, average='micro')
      recall = metrics.recall_score(y_test, y_pred, average='micro')
      f1 = metrics.f1_score(y_test, y_pred, average='micro')
      #auroc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
      confmat = metrics.confusion_matrix(y_test, y_pred)
      acc = metrics.accuracy_score(y_test, y_pred)
      precision = metrics.precision_score(y_test, y_pred, average='micro')
      recall = metrics.recall_score(y_test, y_pred, average='micro')
      f1 = metrics.f1_score(y_test, y_pred, average='micro')
      #auroc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
      if printResults:      
            print(f"Accuracy score: {acc:.5f}\n" + f"Recall: {recall:.5f}\n" + f"F1 score: {f1:.5f}\n")
            fig = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=rf.classes_)
            fig.plot()
            # save figure to PNG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"PNG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "randomForestConfusionMatrix_"+ExperimentDir+"_estimators_"+str(nEstimators)+"_depth_"+str(maxDepth)+".png"
            fig.figure_.savefig(join(DIR,FILE))
            # save figure to SVG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"SVG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "randomForestConfusionMatrix_"+ExperimentDir+"_estimators_"+str(nEstimators)+"_depth_"+str(maxDepth)+".svg"
            fig.figure_.savefig(join(DIR,FILE))
            
      if showModel:
            # INTENTO DE PINTAR EL ÁRBOL DE DECISIÓN
            count=0
            for tree_in_forest in rf.estimators_:
                  count=count+1
                  fig = plt.figure(figsize=(30,10)) #Para cambiar el tamaño del árbol
                  tree.plot_tree(tree_in_forest, max_depth=3, feature_names=X_train.columns, fontsize= 11, filled=True)
                  # save figure to PNG file
                  DIR= join(outputPath,ExperimentDir,imagesDir,"PNG")
                  #create directory if not exists
                  if not os.path.exists(DIR):
                        os.makedirs(DIR)
                  FILE= "randomForestGeneratedTree_"+str(count)+"_"+ExperimentDir+"_estimators_"+str(nEstimators)+"_depth_"+str(maxDepth)+".png"
                  fig.savefig(join(DIR,FILE))
                  # save figure to SVG file
                  DIR= join(outputPath,ExperimentDir,imagesDir,"SVG")
                  #create directory if not exists
                  if not os.path.exists(DIR):
                        os.makedirs(DIR)
                  FILE= "decisionTreeGeneratedTree_"+str(count)+"_"+ExperimentDir+"_estimators_"+str(nEstimators)+"_depth_"+str(maxDepth)+".svg"
                  fig.savefig(join(DIR,FILE))
                  
      if saveResults:
            #save results
            #results = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
            DIR=join(outputPath,ExperimentDir,resultsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            #FILE="decisionTreeResults_"+ExperimentDir+"_depth_"+str(maxDepth)+".csv"
            #results.to_csv(join(DIR,FILE))
            # create txt with accuracy and recall
            FILE="randomForestResults_"+ExperimentDir+"_estimators_"+str(nEstimators)+"_depth_"+str(maxDepth)+".txt"
            with open(join(DIR,FILE), "w") as text_file:
                  #write in text file
                  text_file.write(f"Confusion matrix:\n{confmat}\n" +
                  f"Accuracy score: {acc:.5f}\n" +
                  f"Precision: {precision:.5f}\n"+
                  f"Recall: {recall:.5f}\n" +
                  f"F1 score: {f1:.5f}\n")


      if saveModel:
            # save model
            DIR=join(outputPath,ExperimentDir,modelsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE="randomForestModel_"+ExperimentDir+"_estimators_"+str(nEstimators)+"_depth_"+str(maxDepth)+".joblib"
            dump(rf, join(DIR,FILE))


# *****************************************************************************************
# ** SVM Algorithm
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT:
# ******* OUTPUT: 
# *****************************************************************************************
def svm(maxIter,ker,ExperimentDir,X_train, y_train, X_test, y_test, outputPath,imagesDir,modelsDir,resultsDir, printResults, saveResults, saveModel):
      svm = SVC(random_state=0,max_iter=maxIter,kernel=ker)
      svm.fit(X_train, y_train)
      #svm=SVC(kernel='linear', C=1).fit(X_train, y_train)
      y_pred = svm.predict(X_test)
      
      #ANALISIS DE RESULTADOS
      confmat = metrics.confusion_matrix(y_test, y_pred)
      acc = metrics.accuracy_score(y_test, y_pred)
      precision = metrics.precision_score(y_test, y_pred, average='micro')
      recall = metrics.recall_score(y_test, y_pred, average='micro')
      f1 = metrics.f1_score(y_test, y_pred, average='micro')
      #auroc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
      if printResults:
            print(f"Accuracy score: {acc:.5f}\n" + f"Recall: {recall:.5f}\n" + f"F1 score: {f1:.5f}\n")
            fig = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=svm.classes_)
            fig.plot()
            # save figure to PNG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"PNG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "svmConfusionMatrix_"+ExperimentDir+"_"+ker+"_iter_"+str(maxIter)+".png"
            fig.figure_.savefig(join(DIR,FILE))
            # save figure to SVG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"SVG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "svmConfusionMatrix_"+ExperimentDir+"_"+ker+"_iter_"+str(maxIter)+".svg"
            fig.figure_.savefig(join(DIR,FILE))
            
      if saveResults:
            #save results
            DIR=join(outputPath,ExperimentDir,resultsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            # create txt with accuracy and recall
            FILE="svmResults_"+ExperimentDir+"_"+ker+"_iter_"+str(maxIter)+".txt"
            with open(join(DIR,FILE), "w") as text_file:
                  text_file.write(f"Confusion matrix:\n{confmat}\n" +
                  f"Accuracy score: {acc:.5f}\n" +
                  f"Precision: {precision:.5f}\n"+
                  f"Recall: {recall:.5f}\n" +
                  f"F1 score: {f1:.5f}\n")
      if saveModel:
            # save model
            DIR=join(outputPath,ExperimentDir,modelsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE="svmModel_"+ExperimentDir+"_"+ker+"_iter_"+str(maxIter)+".joblib"
            dump(svm, join(DIR,FILE))


# *****************************************************************************************
# ** KNN Algorithm
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT:
# ******* OUTPUT: 
# *****************************************************************************************
def knn(nNeighbors,ExperimentDir,X_train, y_train, X_test, y_test, outputPath,imagesDir,modelsDir,resultsDir, printResults, showModel, saveResults, saveModel):
      knn = KNeighborsClassifier(n_neighbors=nNeighbors)
      knn.fit(X_train, y_train)
      y_pred = knn.predict(X_test)

      #ANALISIS DE RESULTADOS
      confmat = metrics.confusion_matrix(y_test, y_pred)
      acc = metrics.accuracy_score(y_test, y_pred)
      precision = metrics.precision_score(y_test, y_pred, average='micro')
      recall = metrics.recall_score(y_test, y_pred, average='micro')
      f1 = metrics.f1_score(y_test, y_pred, average='micro')
      #auroc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
      if printResults:
            print(f"Accuracy score: {acc:.5f}\n" + f"Recall: {recall:.5f}\n" + f"F1 score: {f1:.5f}\n")
            fig = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=knn.classes_)
            fig.plot()
            # save figure to PNG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"PNG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "knnConfusionMatrix_"+ExperimentDir+"_depth_"+str(nNeighbors)+".png"
            fig.figure_.savefig(join(DIR,FILE))
            # save figure to SVG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"SVG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "knnConfusionMatrix_"+ExperimentDir+"_depth_"+str(nNeighbors)+".svg"
            fig.figure_.savefig(join(DIR,FILE))
            
      if saveResults:
            #save results
            DIR=join(outputPath,ExperimentDir,resultsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            # create txt with accuracy and recall
            FILE="knnResults_"+ExperimentDir+"_depth_"+str(nNeighbors)+".txt"
            with open(join(DIR,FILE), "w") as text_file:
                  text_file.write(f"Confusion matrix:\n{confmat}\n" +
                  f"Accuracy score: {acc:.5f}\n" +
                  f"Precision: {precision:.5f}\n"+
                  f"Recall: {recall:.5f}\n" +
                  f"F1 score: {f1:.5f}\n")
      if saveModel:
            # save model
            DIR=join(outputPath,ExperimentDir,modelsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE="knnModel_"+ExperimentDir+"_depth_"+str(nNeighbors)+".joblib"
            dump(knn, join(DIR,FILE))


# *****************************************************************************************
# ** MLP Algorithm
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT:
# ******* OUTPUT: 
# *****************************************************************************************
def mlp(hiddenLayers,activationFunc,maxIter,ExperimentDir,X_train, y_train, X_test, y_test, outputPath,imagesDir,modelsDir,resultsDir, printResults, saveResults, saveModel):
      mlp = MLPClassifier(hidden_layer_sizes=hiddenLayers,activation=activationFunc,max_iter=maxIter)
      #mlp=MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter = 300,activation = 'relu', solver = 'adam')
      mlp.fit(X_train, y_train)
      y_pred = mlp.predict(X_test)
      y_pred_proba = mlp.predict_proba(X_test)#[:, 1]
      #ANALISIS DE RESULTADOS
      confmat = metrics.confusion_matrix(y_test, y_pred)
      acc = metrics.accuracy_score(y_test, y_pred)
      precision = metrics.precision_score(y_test, y_pred, average='micro')
      recall = metrics.recall_score(y_test, y_pred, average='micro')
      f1 = metrics.f1_score(y_test, y_pred, average='micro')
      #auroc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
      #create a string with the hidden layers
      hiddenLayersStr=activationFunc
      for i in range(len(hiddenLayers)):
            hiddenLayersStr=hiddenLayersStr+"_"+str(hiddenLayers[i])
      if printResults:
            print(f"Accuracy score: {acc:.5f}\n" + f"Recall: {recall:.5f}\n" + f"F1 score: {f1:.5f}\n")
            fig = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=mlp.classes_)
            fig.plot()
            # save figure to PNG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"PNG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "mlpConfusionMatrix_"+ExperimentDir+"_"+hiddenLayersStr+"_iter_"+str(maxIter)+".png"
            fig.figure_.savefig(join(DIR,FILE))
            # save figure to SVG file
            DIR = join(outputPath,ExperimentDir,imagesDir,"SVG")
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE= "mlpConfusionMatrix_"+ExperimentDir+"_"+hiddenLayersStr+"_iter_"+str(maxIter)+".svg"
            fig.figure_.savefig(join(DIR,FILE))
            
      if saveResults:
            #save results
            DIR=join(outputPath,ExperimentDir,resultsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            # create txt with accuracy and recall
            FILE="mlpResults_"+ExperimentDir+"_"+hiddenLayersStr+"_iter_"+str(maxIter)+".txt"
            with open(join(DIR,FILE), "w") as text_file:
                  text_file.write(f"Confusion matrix:\n{confmat}\n" +
                  f"Accuracy score: {acc:.5f}\n" +
                  f"Precision: {precision:.5f}\n"+
                  f"Recall: {recall:.5f}\n" +
                  f"F1 score: {f1:.5f}\n")
      if saveModel:
            # save model
            DIR=join(outputPath,ExperimentDir,modelsDir)
            #create directory if not exists
            if not os.path.exists(DIR):
                  os.makedirs(DIR)
            FILE="mlpModel_"+ExperimentDir+"_"+hiddenLayersStr+"_iter_"+str(maxIter)+".joblib"
            dump(mlp, join(DIR,FILE))

# *****************************************************************************************
# ** Decision tree Algorithm - load an existing model
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT: 
# *******  INPUT:
# ******* OUTPUT: 
# *****************************************************************************************
