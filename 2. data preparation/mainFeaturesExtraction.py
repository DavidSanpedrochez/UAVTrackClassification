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

# *****************************************************************************************
# ** Load data stored in a csv file
# *******  INPUT: CSV file path
# ******* OUTPUT: Dataframe with the data
# *****************************************************************************************
def loadDataCSV(CSVList, debug=False):
    csv = pd.read_csv(CSVList,sep=',')
    if debug:
        print("trajectory loaded")
        csv
    return csv

#############################################################
#                        FLAGS                              #
#############################################################
#activate processes flags
processResampling_BeforeFeatures = True
processResampling_AfterFeatures  = False
paralelize          = True
numThreads          = 16 # Number of threads to use
executeWIthPArtialFeatures = True
# Output flags
debug             = False
savePartialInputs = True

#############################################################
#                        PATHS                              #
#############################################################
#Input PATHS
inputPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "FeaturesExtraction","Input")               # PATH to the folder with input data
tracksDir        = "SingleTrackFiles"    # PATH to the sub-folder with input trajectories
instancesCSVFile = 'Dataset final.csv'   # File with the list of trajectories to be processed
segmentsCSVFile  = "segmentsCSVList.csv" # File with the list of segments     to be processed
#Output PATHS
outputPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "FeaturesExtraction")               # PATH to the folder with output data
dividedThreadsDir     = "DividedThreads"          # PATH to the folder to generate output data
resamplingSegmentsDir = "ResamplingSegmentOutput" # PATH to the sub-folder with pre-processed inputs
resamplingTracksDir   = "ResamplingTracksOutput"  # PATH to the sub-folder with pre-processed inputs
segmentationDir       = "SegmentationOutput"      # PATH to the sub-folder with pre-processed inputs
featuresDir           = "FeaturesOutput"          # PATH to the sub-folder with pre-processed inputs

#############################################################
#                     FUNCTIONS                             #
#############################################################


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
    instancesCSVFile = os.path.join(inputPath, instancesCSVFile)
    instancesCSV = loadDataCSV(instancesCSVFile)
    # Separación del dataset en train y test, 80 y 20%
    if executeWIthPArtialFeatures:
        subfiles= os.path.join(outputPath,dividedThreadsDir)
        # for each file in subfiles directory
        for file in os.listdir(subfiles):
            # if file is a csv file
            if file.endswith(".csv"):
                # load the file
                procesedFiles = loadDataCSV(os.path.join(subfiles,file))
                procesedTracks  = procesedFiles['track_id'].unique()
                for track in procesedTracks:
                    # get index of the ULG in instancesCSV
                    index = instancesCSV[instancesCSV['ULG'] == track].index
                    # delete the track from instancesCSV
                    instancesCSV = instancesCSV.drop(index)
        
        # randomize X and y in the same order       
        instancesCSV = instancesCSV.sample(frac=1).reset_index(drop=True)
        #reset index
        instancesCSV = instancesCSV.reset_index(drop=True)
        #      
        
    X = instancesCSV['ULG']
    y = instancesCSV['General']

    # Read segments CSV
    segmentsCSVFile = os.path.join(inputPath, segmentsCSVFile)
    segmentsCSV = loadDataCSV(segmentsCSVFile)
    # Split Xy
    segmentsCSV['seg_id_unique'] = segmentsCSV['track_id']+"_"+segmentsCSV['seg_id'].astype(str) # Add seg_id_unique with track_id and seg_id
    #X = segmentsCSV['seg_id_unique']
    #y = segmentsCSV['UAV_Airframe']

    # execute data balancing
    #if processResampling:
    #    dir = join(outputPath, resamplingTracksDir)
    #    X,y = dataBalancer(X, y,  debug,seed=0)   

    # paralellize execution
    if paralelize:
        if numThreads<1:
            numThreads = psutil.cpu_count()     
        numFiles = len(X) # Number of files to process
        numFilesPerThread = int(numFiles/numThreads) # Number of files to process per thread
        print("Number of files to process: ", numFiles)
        print("Number of files per thread: ", numFilesPerThread)
        threads = []
        for i in range(numThreads):
            # get files to process in this thread
            if i == numThreads-1:  # last thread processes the remaining files
                filesToProcess   = X[i*numFilesPerThread:]
                associatedModels = y[i*numFilesPerThread:]
            else:                  # other threads process the files assigned to them
                filesToProcess   = X[i*numFilesPerThread:(i+1)*numFilesPerThread] 
                associatedModels = y[i*numFilesPerThread:(i+1)*numFilesPerThread]
            # Create the thread
            tracksDirPath  = join(inputPath, tracksDir)
            outputPathPath = join(outputPath, dividedThreadsDir)
            t = multiprocessing.Process(target=featuresExtraction, args=(tracksDirPath, outputPathPath, i, filesToProcess, associatedModels, segmentsCSV))
            threads.append(t)
            t.start()
        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Merge the results
        featuresList = pd.DataFrame()
        for i in range(numThreads):
            filename = os.path.join(outputPath,dividedThreadsDir, f"features_{i}.csv")
            # read the CSV with Pandas
            df = pd.read_csv(filename, header=0)
            featuresList = pd.concat([featuresList, df], axis=0)
            if i == 0:
                # set header to CombinedDF
                featuresList.columns = df.columns
            # Remove the individual files
            os.remove(filename)

        # Save the combined dataframe
        csvFile=join(outputPath,featuresDir,"featuresCSVList.csv")
        if savePartialInputs:
            featuresList.to_csv(csvFile,index=False)

    # If not paralellize, execute sequentially
    else:
        tracksDirPath  = join(inputPath, tracksDir)
        outputPathPath = join(outputPath, dividedThreadsDir)
        featuresList = featuresExtraction(tracksDirPath, outputPathPath, 0, X, y, segmentsCSV)
        if savePartialInputs:
            filename = os.path.join(outputPath,featuresDir, f"features_0.csv")
            # read the CSV with Pandas
            featuresList = pd.read_csv(filename, header=0)
            os.remove(filename)
            featuresList.to_csv(join(outputPath,featuresDir,"featuresCSVList.csv"), index=False)


    # Separate features from labels 
    y = featuresList['UAV_Airframe']
    X = featuresList.drop(['UAV_Airframe'], axis=1)

    # execute data balancing
    if processResampling_AfterFeatures:
        # Data must be reshaped to be able to use the dataBalancer function
        reshapedX=np.array(X).reshape(-1,1)
        reshapedY=np.array(y).reshape(-1,1)#reshapedY=np.array(y)
        dir = join(outputPath,resamplingSegmentsDir)
        X,y = dataBalancer(reshapedX, reshapedY,  debug,dir,outputPath,seed=0)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
    # save data to files
    X_train.to_csv(join(outputPath, featuresDir, "X_train.csv"), index=False)
    X_test.to_csv( join(outputPath, featuresDir, "X_test.csv"),  index=False)
    y_train.to_csv(join(outputPath, featuresDir, "y_train.csv"), index=False)
    y_test.to_csv( join(outputPath, featuresDir, "y_test.csv"),  index=False) 