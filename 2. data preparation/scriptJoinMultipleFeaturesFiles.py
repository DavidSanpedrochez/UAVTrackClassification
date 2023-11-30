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
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split        # To split the data into train and test
from os.path import isfile, join                            # To use the data files
import os
import imblearn                                             # To balance the data
import psutil                                               # To copy files                
import multiprocessing                                      # for parallelization
import pathlib
##### Algorithms to import #####
import segmentationAlgorithms as segmentation               # segmentationAlgorithms.py - To segment the data
#from ..utils.loadCSV import loadDataCSV
from balanceDataset import dataBalancer
from featuresExtraction import featuresExtraction

import pandas as pd



#############################################################
#                        FLAGS                              #
#############################################################
#activate processes flags
processResampling_AfterFeatures=True
explainBalancer=False

#############################################################
#                        PATHS                              #
#############################################################
#Input PATHS
inputPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "FeaturesExtraction","Input")               # PATH to the folder with input data
filesDir        = "DividedThreads"    # PATH to the sub-folder with input trajectories
featuresCSVFile  = "featuresCSVList.csv" # File with the list of segments     to be processed
#Output PATHS
outputPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "FeaturesExtraction")               # PATH to the folder with output data
featuresDir           = "FeaturesOutput"          # PATH to the sub-folder with pre-processed inputs
resamplingSegmentsDir= "ResamplingSegmentOutput"
#############################################################
#                     FUNCTIONS                             #
#############################################################
# *****************************************************************************************
# ** Load data stored in a csv file
# *******  [INPUT] CSV file path
# *******  [OUTPUT] Dataframe with the data
# *****************************************************************************************
def loadDataCSV(CSVList, debug=False):
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
    parser.add_argument('--useFlags',       type=str, default="No",      help='Yes/No')
    parser.add_argument('--Debug',          type=str, default="No",      help='Yes/No')
    parser.add_argument('--Resampling',     type=str, default="No",      help='Before/After/No')
    parser.add_argument('--Segmentation',   type=str, default='SQUISHE', help='Desired algorithm SQUISHE')
    parser.add_argument('--CRate',          type=str, default="50",      help='Int with the desired porcentage of selected')
    parser.add_argument('--MinPoints',      type=int, default="10",      help='Int with the desired number')
    parser.add_argument('--Parallel',       type=str, default="No",      help='Yes/No')
    parser.add_argument('--Threads',        type=int, default="4",       help='Int with the desired number of threads, -1 to auto calculate')
    parser.add_argument('--partialInputs',  type=str, default="Yes",     help='Yes/No')
    args = parser.parse_args()

    # Algorithm configuration
    print("Using arguments")
    processSegmentation = args.Segmentation
    if args.Debug == "Yes": debug = True
    else:                   debug = False
    if args.Resampling == "Before": processResampling = True
    else:                           processResampling = False
    if args.Parallel == "Yes":
        paralelize = True
        numThreads = args.Threads # Number of threads to use
    else:
        paralelize = False
    # Output flags
    if args.partialInputs == "Yes":
        savePartialInputs=True
    else:
        savePartialInputs=False
    
    ### Execute process
    # Read input data
    if debug:
        print("DEBUG: Loading data...")
    
    # Read CSV
    instancesCSVFile = os.path.join(inputPath, featuresCSVFile)
    featuresList = loadDataCSV(instancesCSVFile)
    # Separación del dataset en train y test, 80 y 20%
    
    partialFeaturesDir= os.path.join(inputPath, filesDir) 
    #for each file in the directory
    for file in os.listdir(partialFeaturesDir):
        if file.endswith(".csv"):
            #read the file
            newFile=loadDataCSV(os.path.join(partialFeaturesDir,file))
            featuresList = featuresList.append(newFile)
            if debug:
                print("DEBUG: Loading data...")
                featuresList
   
    
    featuresList.to_csv(join(outputPath,featuresDir,"ALLfeaturesCSVList.csv"), index=False)


    # Separate features from labels 
    y = featuresList['UAV_Airframe']
    X = featuresList.drop(['UAV_Airframe'], axis=1)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
    # save data to files
    #X_train.to_csv(join(outputPath, featuresDir, "X_train_unbalanced.csv"), index=False)
    #X_test.to_csv( join(outputPath, featuresDir, "X_test_unbalanced.csv"),  index=False)
    #y_train.to_csv(join(outputPath, featuresDir, "y_train_unbalanced.csv"), index=False)
    #y_test.to_csv( join(outputPath, featuresDir, "y_test_unbalanced.csv"),  index=False)

    # execute data balancing
    processResampling_AfterFeatures=True
    if processResampling_AfterFeatures:
        # Data must be reshaped to be able to use the dataBalancer function
        #reshapedX=np.array(X).reshape(-1,1)
        #reshapedY=np.array(y).reshape(-1,1)#reshapedY=np.array(y)
        dir = join(outputPath,"ResamplingSegmentOutput")
        X,y = dataBalancer(X, y,  debug,explainBalancer,dir,seed=0)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
    # save data to files
    X_train.to_csv(join(outputPath, featuresDir, "X_train.csv"), index=False)
    X_test.to_csv( join(outputPath, featuresDir, "X_test.csv"),  index=False)
    y_train.to_csv(join(outputPath, featuresDir, "y_train.csv"), index=False)
    y_test.to_csv( join(outputPath, featuresDir, "y_test.csv"),  index=False) 