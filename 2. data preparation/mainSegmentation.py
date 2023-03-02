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
import shutil
##### Algorithms to import #####
import segmentationAlgorithms as segmentation               # segmentationAlgorithms.py - To segment the data 

#############################################################
#                        FLAGS                              #
#############################################################
#activate processes flags
processResampling_Beforesegments = True         # Resampling before segments extraction 
processResampling_Aftersegments  = False        #         
processSegmentation              = "SQUISHE"    #             
compressionRate                  = 1/100        #         
minCompressionLen                = 10           #     
paralelize                       = True         #     
numThreads                       = 16           # Number of threads to use
# Output flags
debug             = False
savePartialInputs = True

#############################################################
#                        PATHS                              #
#############################################################
#Input PATHS
inputPath        = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "Segmentation","Input")               # PATH to the folder with input data
tracksDir        = "SingleTrackFiles"             # PATH to the sub-folder with input tracks
instancesCSVFile = 'Dataset final.csv'   # File with the list of instances to be processed
#Output PATHS
outputPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data" , "Segmentation")               # PATH to the folder with output data
dividedThreadsDir     = "DividedThreads"          # PATH to the folder to generate output data
resamplingSegmentsDir = "ResamplingSegmentOutput" # PATH to the sub-folder with pre-processed inputs
resamplingTracksDir   = "ResamplingTracksOutput"  # PATH to the sub-folder with pre-processed inputs
segmentationDir       = "SegmentationOutput"      # PATH to the sub-folder with pre-processed inputs

#############################################################
#                     FUNCTIONS                             #
#############################################################
# *****************************************************************************************
# ** Load data stored in a csv file
# *******  INPUT: CSV file path
# ******* OUTPUT: Dataframe with the data
# *****************************************************************************************
def loadDataCSV(CSVList):
    csv = pd.read_csv(CSVList,sep=',')
    if debug:
        print("trajectory loaded")
        csv
    return csv

#####################################
# This function extracts the trajectory summary from the ulog file
def extractSegmentSummary(track=None, segmentStart=None, segmentEnd=None):
    segmentStart = int(segmentStart)
    segmentEnd   = int(segmentEnd)
    
    # initialize variables to store the summary
    # calculate duration in seconds
    duration       = (track.timestamp[segmentEnd] - track.timestamp[segmentStart]) / 1000000 
    numPoints      = segmentEnd - segmentStart
    distanceXY_sum = 0
    distanceZ_sum  = 0
    speedXY_sum    = 0
    speedZ_sum     = 0

    # iterate over all data
    for i in range(segmentStart, segmentEnd):
        if i != 0:
            distanceXY_sum += math.sqrt((track.x[i] - track.x[i-1])**2 + (track.y[i] - track.y[i-1])**2)
            distanceZ_sum  += abs(track.z[i] - track.z[i-1])
            speedXY_sum    += math.sqrt((track.x[i] - track.x[i-1])**2 + (track.y[i] - track.y[i-1])**2) / (track.timestamp[i] - track.timestamp[i-1])
            speedZ_sum     += abs(track.z[i] - track.z[i-1]) / (track.timestamp[i] - track.timestamp[i-1])
    # Calculate speed average
    speedXY_sum = speedXY_sum / numPoints
    speedZ_sum  = speedZ_sum / numPoints
    return duration, numPoints, distanceXY_sum, distanceZ_sum, speedXY_sum, speedZ_sum

# *****************************************************************************************
# Segment the tracks
# *******  INPUT: tracksDir:        path to the folder with the tracks
# *******  INPUT: outputPath:       path to the folder to store the segments and its summary
# *******  INPUT: flag              to indicate if segmentation can be applied
# *******  INPUT: thread_id:        thread  number
# *******  INPUT: tracks:           list of tracks to be processed 
# *******  INPUT: models:           drone model for each track
# ******* OUTPUT: segmentAlgorithm: CSV file with the dataframe including the segments extracted from the tracks
# *****************************************************************************************
def segmentationProcess(tracksDir, outputPath, thread_id, tracks, models, segmentAlgorithm="SQUISHE", debug=False):
    # Remove output folder if exists and create it again
    #if os.path.exists(outputPath):
    #    shutil.rmtree(outputPath)    
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    tracks=tracks.reset_index(drop=True)
    models=models.reset_index(drop=True)
    if debug:
        print("thread_id")
        print(thread_id)
    df_segments = pd.DataFrame(
        columns=[
            'UAV_Airframe',
            'track_id',
            'segmentStart',
            'segmentEnd',
            'seg_id',
            'num_segments',
            'duration',
            'numPoints',
            'distanceXY',
            'distanceZ',
            'speedXY',
            'speedZ'
        ]
    )   

    # for each track in x_train and model in y_train
    if debug:
        print(tracks)
    for i in range(len(tracks)): # for each track
        try:
            currentDroneModel = models[i] # Get drone model name
            trackFile         = tracks[i]+".csv"
            trackID           = tracks[i]
        except:
            print(f"Error reading file number {i} of {len(tracks)}")

        # Print the percentage of tracks processed each 10 tracks
        if i%10==0:
            print("Thread "+str(thread_id)+" has processed "+str(i)+" tracks of "+str(len(tracks))+" ("+str(round(i/len(tracks)*100,2))+"%)")

        pathCSVSegments = os.path.join(outputPath, f"segments_{thread_id}.csv")

        try:
            # Read file
            trackPathFile = join(tracksDir,trackFile)
            if isfile(trackPathFile):
                df = pd.read_csv(trackPathFile,sep=',')  
                
                # Select segmentation algorithm
                if segmentAlgorithm=="SQUISHE":
                    indexList=segmentation.SQUISHE(df, compressionRate, minCompressionLen,thread_id,debug) #create segments
                else:
                    pass # TO EXPAND IN THE FUTURE
                
                # On each segment, store on a CSV file its data
                segmentID=0
                for i in range(len(indexList)-1):
                    segmentStart = indexList.iloc[i]['index']
                    segmentEnd   = indexList.iloc[i+1]['index']
                    duration, numPoints, distanceXY, distanceZ, speedXY, speedZ = \
                        extractSegmentSummary(track=df, segmentStart=segmentStart, segmentEnd=segmentEnd)

                    # Create a new dataframe with the segment data
                    new_segment = pd.DataFrame(
                        columns=[
                            'UAV_Airframe',
                            'track_id',
                            'segmentStart',
                            'segmentEnd',
                            'seg_id',
                            'num_segments',
                            'duration',
                            'numPoints',
                            'distanceXY',
                            'distanceZ',
                            'speedXY',
                            'speedZ'
                        ]
                    )
                    new_segment.at[0,'UAV_Airframe'] = currentDroneModel
                    new_segment.at[0,'track_id']     = trackID
                    new_segment.at[0,'segmentStart'] = segmentStart
                    new_segment.at[0,'segmentEnd']   = segmentEnd
                    new_segment.at[0,'seg_id']       = i
                    new_segment.at[0,'num_segments'] = len(indexList)-1
                    new_segment.at[0,'duration']     = duration         
                    new_segment.at[0,'numPoints']    = numPoints
                    new_segment.at[0,'distanceXY']   = distanceXY
                    new_segment.at[0,'distanceZ']    = distanceZ
                    new_segment.at[0,'speedXY']      = speedXY
                    new_segment.at[0,'speedZ']       = speedZ
                    
                    segmentID=segmentID+1

                    # Add the new segment to the list of segments
                    df_segments=pd.concat([df_segments, new_segment], ignore_index=True)

                    # Save the new segment on the file of this thread (overwrite all the time)
                    df_segments.to_csv(pathCSVSegments, index=False)

        except Exception as err:
            print(f"Error en thread {thread_id}, ejecutando la iteración {i} del fichero {trackFile}")
            print(err)
    
    # Final storing of the segments
    df_segments.to_csv(pathCSVSegments, index=False)


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
    if args.Debug == "Yes":
        debug = True
    else:
        debug = False
    if args.Resampling == "Before":
        processResampling_Beforesegments = True
        processResampling_Aftersegments  = False
    elif args.Resampling == "After":
        processResampling_Beforesegments = False
        processResampling_Aftersegments  = True
    else:
        processResampling_Beforesegments = False
        processResampling_Aftersegments  = False
    processSegmentation= args.Segmentation 
    if args.Parallel == "Yes":
        paralelize = True
        numThreads = 16 # Number of threads to use
    else:
        paralelize = False
    # Output flags
    if args.partialInputs == "Yes": savePartialInputs=True
    else:                           savePartialInputs=False
    
    ### Execute process
    # Read input data
    if debug:
        print("DEBUG: Loading data...")
    
    # Read CSV
    pathsCSV = join(inputPath,instancesCSVFile)
    instancesCSV = loadDataCSV(pathsCSV)

    # Split data from class
    X = instancesCSV['ULG']
    y = instancesCSV['General']
       
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
            if i == numThreads-1: 
                filesToProcess   = X[i*numFilesPerThread:]        # last thread processes the remaining files
                associatedModels = y[i*numFilesPerThread:]
            else: 
                filesToProcess   = X[i*numFilesPerThread:(i+1)*numFilesPerThread] # other threads process the files assigned to them
                associatedModels = y[i*numFilesPerThread:(i+1)*numFilesPerThread]
            # Create the thread
            tracksDirPath  = join(inputPath,tracksDir)
            outputPathPath = join(outputPath,dividedThreadsDir)
            t = multiprocessing.Process(target=segmentationProcess, args=(tracksDirPath, outputPathPath, i, filesToProcess, associatedModels, processSegmentation))
            threads.append(t)
            t.start()
        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Merge the results
        segmentsList = pd.DataFrame()
        for i in range(numThreads):
            filename = os.path.join(outputPath, dividedThreadsDir, f"segments_{i}.csv")
            # read the CSV with Pandas
            df = pd.read_csv(filename, header=0)
            segmentsList = pd.concat([segmentsList, df], axis=0)
            if i == 0:
                # set header to CombinedDF
                segmentsList.columns = df.columns
            # Remove the individual files
            os.remove(filename)

        # Save the combined dataframe
        csvFile=join(outputPath, segmentationDir, "segmentsCSVList.csv")
        if savePartialInputs: segmentsList.to_csv(csvFile,index=False)

    # Single thread execution
    else:
        tracksDirPath  = join(inputPath, tracksDir)
        outputPathPath = join(outputPath, segmentationDir)
        segmentsList = segmentationProcess(tracksDir=tracksDirPath, outputPath=outputPathPath, thread_id=0, tracks=X, models=y,
                                         segmentAlgorithm=processSegmentation, debug=debug)
        
        # Save the combined dataframe
        if savePartialInputs:
            filename = os.path.join(outputPath,segmentationDir, f"segments_0.csv")
            # read the CSV with Pandas
            segmentsList = pd.read_csv(filename, header=0)
            os.remove(filename)
            segmentsList.to_csv(join(outputPath,segmentationDir,"segmentsCSVList.csv"), index=False)

    # Save the combined dataframe
