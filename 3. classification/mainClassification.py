################################
# This script is the main script of the UAV Track Classification project.
# 0. Include all requirements
# 1. Configure the flags
# 2. Configure the input and output paths
# 3. Run the script
# 
# @author: David Sanchez <davsanch@inf.uc3m.es>
# @author: Daniel Amigo <damigo@inf.uc3m.es>
################################
#############################################################
#                       IMPORTS                             #
#############################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split        # To split the data into train and test
from os.path import isfile, join                            # To use the data files
import psutil                                               # To copy files                
from os.path import isfile, join                            # To use the data files
import os
import pathlib
from matplotlib import pyplot as plt
##### Algorithms to import #####
import classificationAlgorithms as classification           # classificationtionAlgorithms.py - To clasify the data 
#############################################################
#                        FLAGS                              #
#############################################################
 
#activate processes flags
debug         = True
printResults  = True
loadModel     = False
saveResults   = True
saveModel     = True
showModel     = True
algorithmList =['DT']#['DT','RF','KNN','SVM','MLP']
maxDepth      =5
nEstimators   = 100
nNeighbors    = 5
kernel        = 'rbf'
hiddenLayers   = (100)
activationFunc = 'relu'
maxIter        = 1000

#############################################################
#                        PATHS                              #
#############################################################
#Input PATHS
inputPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "Classification","Input") # PATH to the folder with input data
featuresDir = "FeaturesOutput"           # PATH to the sub-folder with input tracks
featuresCSVFile = 'ALLfeaturesCSV.csv'      # File with the list of instances to be processed
#Output PATHS
outputPath            = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "Classification", "Output") # PATH to the folder with output data
imagesDir     = "Figures"          # PATH to the folder to generate output data
modelsDir = "SavedModels" # PATH to the sub-folder with pre-processed inputs
resultsDir   = "Results"  # PATH to the sub-folder with pre-processed inputs

#############################################################
#                     FUNCTIONS                             #
#############################################################
# *****************************************************************************************
# ** Load data stored in a csv file
# *******  [INPUT] CSV file path
# *******  [OUTPUT] Dataframe with the data
# *****************************************************************************************
def loadDataCSV(CSVList):
    csv = pd.read_csv(CSVList,sep=',')
    if debug:
        print("trajectory loaded")
        csv
    return csv

#############################################################
#                        MAIN                               #
#############################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='UAVClassifier')
    parser.add_argument('--useFlags',     type=str, default="No", help='Print debuug messages')
    parser.add_argument('--debug',        type=str, default="No", help='Print debuug messages')
    parser.add_argument('--algorithm',    type=str, default="DT", help='DT/RF/knn/SVM/MLP separated by _')
    parser.add_argument('--loadModel',    type=str, default="No", help='Yes/No')
    parser.add_argument('--saveModel',    type=str, default="No", help='Yes/No')
    parser.add_argument('--printResults', type=str, default="No", help='Yes/No')
    parser.add_argument('--saveResults',  type=str, default="No", help='Yes/No')
    parser.add_argument('--showModel',    type=str, default="No", help='Yes/No')
    
    args = parser.parse_args()
    # Algorithm configuration
    if args.useFlags == "Yes":
        print("Using flags")
    else:
        print("Using arguments")
        #split list to np array
        algorithmList=args.algorithm.split("_")
        if args.debug == "Yes":
            debug = True
        else:
            debug = False
        if args.loadModel == "Yes":
            loadModel = True
        else:
            loadModel = False
        if args.printResults == "printResults":
            printResults = True
        else:
            printResults = False
        if args.saveResults == "Yes":
            saveResults = True
        else:
            saveResults = False
        if args.saveModel == "Yes":
            saveModel = True
        else:
            saveModel = False
        if args.showModel == "Yes":
            showModel = True
        else:
            showModel = False

    #### DEBUG TO DELETE
    # create new dataframe
    sumary = pd.DataFrame(columns=['experiment','Accuracy','Precision','Recall','F1','ConfusionMatrix_00','ConfusionMatrix_11','ConfusionMatrix_22','ConfusionMatrix_33','ConfusionMatrix_01','ConfusionMatrix_02','ConfusionMatrix_03','ConfusionMatrix_10','ConfusionMatrix_12','ConfusionMatrix_13','ConfusionMatrix_20','ConfusionMatrix_21','ConfusionMatrix_23','ConfusionMatrix_30','ConfusionMatrix_31','ConfusionMatrix_32'])
    for directory in os.listdir(join(inputPath,featuresDir)): 
        # execute classification
        #check if is a directory
        if not os.path.isdir(join(inputPath,featuresDir,directory)):
            continue
        #read data from files
        X_train = pd.read_csv(join(inputPath, featuresDir,directory, "X_train.csv"))
        X_test  = pd.read_csv(join(inputPath, featuresDir,directory, "X_test.csv"))
        y_train = pd.read_csv(join(inputPath, featuresDir,directory, "y_train.csv"))
        y_test  = pd.read_csv(join(inputPath, featuresDir,directory, "y_test.csv"))
        
        ### Execute process
        # separate data from labels - extract "segment_id" and "track_id" columns
        X_train_id = X_train[['seg_id_track','seg_id_unique','track_id']]
        X_test_id  = X_test[['seg_id_track','seg_id_unique','track_id']]
        
        # drop "segment_id" and "track_id" columns
        X_train_drop = X_train.drop(['seg_id_track','seg_id_unique','track_id'], axis=1)
        X_test_drop = X_test.drop(['seg_id_track','seg_id_unique','track_id'], axis=1)



        # Read input data
        for algorithm in algorithmList:
            if loadModel:
                if algorithm == 'DT':
                    print("Loading DT model")
                    #TOCOMPLETE
                elif algorithm == 'RF':
                    print("Loading RF model")
                    #TOCOMPLETE
                elif algorithm == 'KNN':
                    print("Loading knn model")
                    #TOCOMPLETE
                elif algorithm == 'SVM':
                    print("Loading SVM model")
                    #TOCOMPLETE
                elif algorithm == 'MLP':
                    print("Loading MLP model")
                    #TOCOMPLETE
            else:
                print("Train new model")
                if algorithm == 'DT':
                    
                    confmat,acc,precision,recall,f1=classification.decisionTree(maxDepth,directory,X_train_drop, y_train, X_test_drop, y_test,outputPath,imagesDir,modelsDir,resultsDir, printResults, showModel, saveResults, saveModel)
                    #save results
                    sumary = sumary.append({'experiment':directory,'Accuracy':acc,'Precision':precision,'Recall':recall,'F1':f1,'ConfusionMatrix_00':confmat[0,0],'ConfusionMatrix_11':confmat[1,1],'ConfusionMatrix_22':confmat[2,2],'ConfusionMatrix_33':confmat[3,3],'ConfusionMatrix_01':confmat[0,1],'ConfusionMatrix_02':confmat[0,2],'ConfusionMatrix_03':confmat[0,3],'ConfusionMatrix_10':confmat[1,0],'ConfusionMatrix_12':confmat[1,2],'ConfusionMatrix_13':confmat[1,3],'ConfusionMatrix_20':confmat[2,0],'ConfusionMatrix_21':confmat[2,1],'ConfusionMatrix_23':confmat[2,3],'ConfusionMatrix_30':confmat[3,0],'ConfusionMatrix_31':confmat[3,1],'ConfusionMatrix_32':confmat[3,2]}, ignore_index=True)
                elif algorithm == 'RF':
                    #TOCOMPLETE
                    classification.randomForest(maxDepth,nEstimators,directory,X_train_drop, y_train, X_test_drop, y_test,outputPath,imagesDir,modelsDir,resultsDir, printResults, showModel, saveResults, saveModel)
                    print("TO DO")
                elif algorithm == 'KNN':
                    #TOCOMPLETE
                    classification.knn(nNeighbors,directory,X_train_drop, y_train, X_test_drop, y_test,outputPath,imagesDir,modelsDir,resultsDir, printResults, showModel, saveResults, saveModel)
                    print("TO DO")
                elif algorithm == 'SVM':
                    #TOCOMPLETE
                    classification.svm(maxIter,kernel,directory,X_train_drop, y_train, X_test_drop, y_test,outputPath,imagesDir,modelsDir,resultsDir, printResults, saveResults, saveModel)
                    print("TO DO")
                elif algorithm == 'MLP':
                    #TOCOMPLETE
                    classification.mlp(hiddenLayers,activationFunc,maxIter,directory,X_train_drop, y_train, X_test_drop, y_test,outputPath,imagesDir,modelsDir,resultsDir, printResults, saveResults, saveModel)
                    print("TO DO")
                #save results to csv file
                sumary.to_csv(join(outputPath,"sumaryResults.csv"), index=False)

    