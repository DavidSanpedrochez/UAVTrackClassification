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
import random
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
import balanceDataset as balancer
from featuresExtraction import featuresExtraction

import pandas as pd




#############################################################
#                        FLAGS                              #
#############################################################
#############################################################
#                        PATHS                              #
#############################################################
#Input PATHS
inputPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "Classification","Input")               # PATH to the folder with input data
featuresCSVFile  = "AllfeaturesCSVList.csv" # File with the list of segments     to be processed
#Output PATHS
outputPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "Classification","Input")               # PATH to the folder with output data
featuresDir           = "FeaturesOutput"          # PATH to the sub-folder with pre-processed inputs
balancerDir           = "BalancerOutput"          # PATH to the sub-folder with pre-processed inputs
#############################################################
#                     FUNCTIONS                             #
#############################################################
def createExperiment(featuresList, outputPath, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType):
    # Read CSV
    
    # Separación del dataset en train y test, 80 y 20% 
    DIR=join(outputPath, featuresDir)
    if not os.path.exists(DIR):
                  os.makedirs(DIR)
    featuresList.to_csv(join(DIR,experiment+"_CSVList.csv"), index=False)
    # Separate features from labels 
    y = featuresList['UAV_Airframe']
    X = featuresList.drop(['UAV_Airframe'], axis=1)
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
    # save data to files
    DIR=join(outputPath, featuresDir,"unbalanced", experiment)
    if not os.path.exists(DIR):
                  os.makedirs(DIR)
    X_train.to_csv(join(DIR, "X_train.csv"), index=False)
    X_test.to_csv( join(DIR, "X_test.csv"),  index=False)
    y_train.to_csv(join(DIR, "y_train.csv"), index=False)
    y_test.to_csv( join(DIR, "y_test.csv"),  index=False) 
    # execute data balancing
    if processResampling:
        # Data must be reshaped to be able to use the dataBalancer function
        #reshapedX=np.array(X).reshape(-1,1)
        #reshapedY=np.array(y).reshape(-1,1)#reshapedY=np.array(y)
        if balancerType=="SMOTE":
               print("TO DO")
        elif balancerType=="RANDOMUNDER":
            X,y = balancer.dataBalancer(X, y, debug,explainBalancer,balancerOutputPath,seed)
        elif balancerType=="RANDOMOVER":
            print("TO DO")
        elif balancerType=="TRACKBASEDRANDOMUNDER":
            X,y = balancer.trackBalancer(X,y,debug,explainBalancer,balancerOutputPath,seed)
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
        # save data to files
        experiment=experiment+"_balancerSeed_"+str(seed)
        DIR=join(outputPath, featuresDir,"balanced", experiment)
        if not os.path.exists(DIR):
                    os.makedirs(DIR)
        X_train.to_csv(join(DIR, "X_train.csv"), index=False)
        X_test.to_csv( join(DIR, "X_test.csv"),  index=False)
        y_train.to_csv(join(DIR, "y_train.csv"), index=False)
        y_test.to_csv( join(DIR, "y_test.csv"),  index=False)

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
#                     EXPERIMENTS                           #
#############################################################
def allFeaturesExperiment(inputPath, outputPath, featuresCSVFile, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType):
    # Read CSV
    instancesCSVFile = os.path.join(inputPath, featuresCSVFile)
    featuresList = loadDataCSV(instancesCSVFile)
    # Separación del dataset en train y test, 80 y 20% 
    createExperiment(featuresList, outputPath, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType)

def onlySegmentFeaturesExperiment(inputPath, outputPath, featuresCSVFile, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType):
    # Read CSV
    instancesCSVFile = os.path.join(inputPath, featuresCSVFile)
    featuresList = loadDataCSV(instancesCSVFile)
    # Select only segment features
    featuresList = featuresList[['totalTime_Segment', 'totalHorDistance_Segment',
            'totalVerDistance_Segment', 'totalDistance_Segment', 'stopRate_Segment',
            'max_horSpeed_Segment', 'max_horAcceleration_Segment','max_horJerk_Segment', 
            'min_horSpeed_Segment', 'min_horAcceleration_Segment', 'min_horJerk_Segment',
            'timeRatedTotal_horDistance_Segment', 'timeRatedTotal_horSpeed_Segment', 'timeRatedTotal_horAcceleration_Segment', 'timeRatedTotal_horJerk_Segment',
            'max_z_Segment', 'min_z_Segment','timeRatedTotal_z_Segment',
            'max_vz_Segment', 'min_vz_Segment','timeRatedTotal_vz_Segment',
            'max_az_Segment', 'min_az_Segment','timeRatedTotal_az_Segment',
            'max_jz_Segment', 'min_jz_Segment','timeRatedTotal_jz_Segment',
            'max_horAngle_Segment', 'max_verAngle_Segment', 'min_horAngle_Segment', 'min_verAngle_Segment',
            'timeRatedTotal_horAngle_Segment', 'timeRatedTotal_verAngle_Segment',
            'max_angleRate_Segment', 'min_angleRate_Segment','timeRatedTotal_angleRate_Segment',
            'UAV_Airframe', 'seg_id_track','seg_id_unique', 'track_id']]
    # Separación del dataset en train y test, 80 y 20% 
    createExperiment(featuresList, outputPath, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType)

def onlyTrackFeaturesExperiment(inputPath, outputPath, featuresCSVFile, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType):
    # Read CSV
    instancesCSVFile = os.path.join(inputPath, featuresCSVFile)
    featuresList = loadDataCSV(instancesCSVFile)
    # Select only track features
    featuresList = featuresList[['totalTime_Track', 'totalHorDistance_Track',
            'totalVerDistance_Track', 'totalDistance_Track', 'stopRate_Track',
            'max_horSpeed_Track', 'max_horAcceleration_Track','max_horJerk_Track', 
            'min_horSpeed_Track', 'min_horAcceleration_Track', 'min_horJerk_Track',
            'timeRatedTotal_horDistance_Track', 'timeRatedTotal_horSpeed_Track', 'timeRatedTotal_horAcceleration_Track', 'timeRatedTotal_horJerk_Track',
            'max_z_Track', 'min_z_Track','timeRatedTotal_z_Track',
            'max_vz_Track', 'min_vz_Track','timeRatedTotal_vz_Track',
            'max_az_Track', 'min_az_Track','timeRatedTotal_az_Track',
            'max_jz_Track', 'min_jz_Track','timeRatedTotal_jz_Track',
            'max_horAngle_Track', 'max_verAngle_Track', 'min_horAngle_Track', 'min_verAngle_Track',
            'timeRatedTotal_horAngle_Track', 'timeRatedTotal_verAngle_Track',
            'max_angleRate_Track', 'min_angleRate_Track','timeRatedTotal_angleRate_Track',
            'UAV_Airframe', 'seg_id_track','seg_id_unique', 'track_id']]
    # Separación del dataset en train y test, 80 y 20% 
    createExperiment(featuresList, outputPath, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType)

def onlySelectedFeaturesExperiment(inputPath, outputPath, featuresCSVFile, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType):
    # Read CSV
    instancesCSVFile = os.path.join(inputPath, featuresCSVFile)
    featuresList = loadDataCSV(instancesCSVFile)
    print("TO DO")
    #   ['totalTime_Segment', 'totalHorDistance_Segment',
    #   'totalVerDistance_Segment', 'totalDistance_Segment', 'stopRate_Segment',
    #   'max_horSpeed_Segment', 'max_horAcceleration_Segment','max_horJerk_Segment', 
    #   'min_horSpeed_Segment','min_horAcceleration_Segment', 'min_horJerk_Segment',
    #   'timeRatedTotal_horDistance_Segment', 'timeRatedTotal_horSpeed_Segment','timeRatedTotal_horAcceleration_Segment','timeRatedTotal_horJerk_Segment', 
    #   'max_z_Segment', 'min_z_Segment','timeRatedTotal_z_Segment', 
    #   'max_vz_Segment', 'min_vz_Segment','timeRatedTotal_vz_Segment', 
    #   'max_az_Segment', 'min_az_Segment', 'timeRatedTotal_az_Segment', 
    #   'max_jz_Segment', 'min_jz_Segment','timeRatedTotal_jz_Segment', 
    #   'max_horAngle_Segment', 'max_verAngle_Segment', 'min_horAngle_Segment', 'min_verAngle_Segment',
    #   'timeRatedTotal_horAngle_Segment', 'timeRatedTotal_verAngle_Segment',
    #   'max_angleRate_Segment', 'min_angleRate_Segment',
    #   'timeRatedTotal_angleRate_Segment', 'totalTime_Track',
    #   'totalHorDistance_Track', 'totalVerDistance_Track',
    #   'totalDistance_Track', 'stopRate_Track', 'max_horSpeed_Track',
    #   'max_horAcceleration_Track', 'max_horJerk_Track', 'min_horSpeed_Track',
    #   'min_horAcceleration_Track', 'min_horJerk_Track',
    #   'timeRatedTotal_horDistance_Track', 'timeRatedTotal_horSpeed_Track',
    #   'timeRatedTotal_horAcceleration_Track', 'timeRatedTotal_horJerk_Track',
    #   'max_z_Track', 'min_z_Track', 'timeRatedTotal_z_Track', 'max_vz_Track',
    #   'min_vz_Track', 'timeRatedTotal_vz_Track', 'max_az_Track',
    #   'min_az_Track', 'timeRatedTotal_az_Track', 'max_jz_Track',
    #   'min_jz_Track', 'timeRatedTotal_jz_Track', 'max_horAngle_Track',
    #   'max_verAngle_Track', 'min_horAngle_Track', 'min_verAngle_Track',
    #   'timeRatedTotal_horAngle_Track', 'timeRatedTotal_verAngle_Track',
    #   'max_angleRate_Track', 'min_angleRate_Track',
    #   'timeRatedTotal_angleRate_Track', 'UAV_Airframe', 'seg_id_track',
    #   'seg_id_unique', 'track_id'],
    # Select only desired features
    
    # Separación del dataset en train y test, 80 y 20% 
    createExperiment(featuresList, outputPath, featuresDir,seed, experiment,processResampling,debug,explainBalancer,balancerOutputPath,balancerType)
#############################################################
#                        FLAGS                              #
#############################################################
#activate processes flags
processResampling=True
debug=False
explainBalancer=False
#############################################################
#                        MAIN                               #
#############################################################
if __name__ == "__main__":
    # create 100 random seeds for resampling
    seed = random.sample(range(1, 1000), 100)
    balancerType=["RANDOMUNDER"]
    
    ### Execute process
    for s in seed:
            balancerOutputPath=join(outputPath, balancerDir,str(s))
            if not os.path.exists(balancerOutputPath):
                        os.makedirs(balancerOutputPath)
            for b in balancerType:
                onlySegmentFeaturesExperiment(inputPath, outputPath, featuresCSVFile, featuresDir,s,"SegmentFeatures",processResampling,debug,explainBalancer,balancerOutputPath,b)
                allFeaturesExperiment(inputPath, outputPath, featuresCSVFile, featuresDir,s,"ALLfeatures",processResampling,debug,explainBalancer,balancerOutputPath,b)
    # Read input data
    if debug:
        print("DEBUG: Loading data...")
    

