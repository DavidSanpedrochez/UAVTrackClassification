################################
# This script 
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
from featuresExtraction import featuresExtraction


# *****************************************************************************************
# ** Return the balanced dataset
# *******  INPUT: X
# *******  INPUT: y
# *******  INPUT: path
# *******  INPUT: debug
# *******  INPUT: saveInputData
# ******* OUTPUT: X
# ******* OUTPUT: y
# *****************************************************************************************
def dataBalancer(X,y,debug,explainBalancer,outputPath,seed):
    if debug:
        print("DEBUG: resampling...")
        print("Rate before resampling")
        print(y.value_counts(normalize=True)*100)
    # Reshape the data to avoid errors
    #X=np.array(X).reshape(-1,1)
    #y=np.array(y).reshape(-1,1)
    ###### TODO: Check type of X and y and transform accordingly
    # Resample the data using the RandomUnderSampler
    x_resampled, y_resampled = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='auto',random_state=seed).fit_resample(X,y)  

    # Reshape the data to avoid errors
    X = x_resampled
    y = y_resampled

  
    if debug:
        print("DEBUG: resampling finished")
        print("Rate after resampling")  
        print(y.value_counts(normalize=True)*100)
    if explainBalancer:
        # create data frame using class as column name
        X_id = X[['seg_id_track','seg_id_unique','track_id']]
        # save X_id in a csv file
        X_id.to_csv(os.path.join(outputPath, "balancer_"+str(seed)+".csv"), index=False)


    # Return the balanced data
    return X,y
# *****************************************************************************************
# ** Return the balanced dataset, ensuring segments from same track
# *******  INPUT: X
# *******  INPUT: y
# *******  INPUT: path
# *******  INPUT: debug
# *******  INPUT: saveInputData
# ******* OUTPUT: X
# ******* OUTPUT: y
# *****************************************************************************************
def trackBalancer(X,y,debug,explainBalancer,outputPath,seed):
    x_resampled = pd.DataFrame()
    y_resampled = pd.DataFrame()
    #Calculate number of segments for minoritary class
    minoritaryClass = y.value_counts().idxmin()
    #for each class
    for i in len(range(y.unique())):
        df=pd.DataFrame()
        # while selected segments under the number of segments of the minoritary class
        while len(df) < y.value_counts()[minoritaryClass]:
            #Select a random track_id from X
            print("a")

            #Select all segments from the selected track_id

            #insert selected in resampled data