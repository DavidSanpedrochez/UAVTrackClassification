################################
# This script contains functions to process a set of ulog files. On each one:
#   - extracts the trajectory summary features
#   - using the features several criteria are applied to decide if the trajectory is valid or not
#  - if the trajectory is valid, it is stored CSV file in a folder of valid trajectories
# All trajectories (valid or not) and its calculated features are stored in a CSV file called "trajectorySummary.csv"
# 
# @author: David Sanchez <davsanch@inf.uc3m.es>
# @author: Daniel Amigo <damigo@inf.uc3m.es>
################################

#############################################################
#                       IMPORTS                             #
#############################################################
# Folders and data management
import os
from os.path import isfile, join
import shutil
import sys
import pandas as pd
import pathlib
# Math and data processing
import numpy as np
import pymap3d as pm
# ULOG imports
from ulog2csv import convert_ulog2csv
from ulog_params import convert_ulog_params
from ulog_info import convert_ulog_info
from ulog_info import show_info
from pyulog.core import ULog
# Error notifications using a private Telegram bot
from ..utils.telegramMessaging import messageTelegramBot

#####################################
# This function extracts the trajectory summary from the ulog file
def extractTrajectorySummary(ulog, outputpath, filenameNoExt):
    desiredData = ["vehicle_local_position"]
    distance3DSum, duration, numPoints = 0, 0, 0
    bigTimeJump = False
    bigPositionJump = False
    notEnoughMovementXAxis = False
    notEnoughMovementYAxis = False
    notEnoughMovementZAxis = False

    for d in ulog.data_list: # iterate over all data
        if d.name in desiredData: # if the data is in the desired data list

            # calculate initial timestamp substracting the first timestamp to all the timestamps
            # we have to do this because on header they're written as 0
            if d.name == "vehicle_local_position":
                ############ SUMMARY VARIABLES ############
                # time duration
                timestamp = pd.to_datetime(d.data['timestamp'], unit='us', utc=True)
                timeDuration = timestamp[len(timestamp)-1] - timestamp[0]
                duration     = timeDuration.seconds
                # number of points
                numPoints = len(d.data['timestamp'])
                # calculate distance traveled in 3D
                for j in range(1,len(timestamp)):
                    vector3Dprev = np.array([d.data['x'][j-1], d.data['y'][j-1], d.data['z'][j-1]])
                    vector3Dcurr = np.array([d.data['x'][j],   d.data['y'][j],   d.data['z'][j]])
                    distance3D = np.linalg.norm(vector3Dcurr - vector3Dprev)
                    distance3DSum += distance3D
                    # Calculate seconds between each pair of measurements
                    timeGap = (timestamp[j].value / 1000000000) - (timestamp[j-1].value / 1000000000)
                    # If time gap is over 10sec, it was a noticeable time gap
                    if timeGap > 10:
                        bigTimeJump = True
                    # If 3Dspeed is over 45m/s, there was a noticeable position jump
                    if timeGap > 0 and distance3D / timeGap > 45:
                        bigPositionJump = True

                # At the end of the trajectory, it has enough movement on, at least, one axis? IMPROVEMENT combine XY axes into one variable
                if (max(d.data['x']) - min(d.data['x'])) < 10:
                    notEnoughMovementXAxis = True
                if (max(d.data['y']) - min(d.data['y'])) < 10:
                    notEnoughMovementYAxis = True
                if (max(d.data['z']) - min(d.data['z'])) < 10:
                    notEnoughMovementZAxis = True
                break

    return  distance3DSum, duration, numPoints, bigTimeJump, bigPositionJump, \
            notEnoughMovementXAxis, notEnoughMovementYAxis, notEnoughMovementZAxis

#####################################
# This function stores the trajectory of the drone in a CSV file
def storeTrajectory(ulog, outputpath, filenameNoExt):
    desiredData = ["vehicle_local_position"]
    for d in ulog.data_list: # iterate over all data
        if d.name in desiredData: # if the data is in the desired data list
            # calculate initial timestamp substracting the first timestamp to all the timestamps
            # we have to do this because on header they're written as 0
            if d.name == "vehicle_local_position":
                #### SAVE TRAJECTORY AS CSV ####
                # create a subset of the data as DataFrame
                timestamp = d.data['timestamp'] - d.data['timestamp'][0]
                df = pd.DataFrame({
                    'timestamp': np.array(timestamp,dtype='float64'),
                    'x': d.data['x'],
                    'y': d.data['y'],
                    'z': d.data['z'],
                    'vx': d.data['vx'],
                    'vy': d.data['vy'],
                    'vz': d.data['vz'],
                })

                # store as CSV
                outputFilename = pathlib.Path(outputpath, filenameNoExt+".csv")
                df.to_csv(f"{outputFilename}", index=False)

#####################################
# Function to execute the ulog commands to generate CSV files from the ULG files
# Writes the drone model from the trajectory from ulog_param in a CSV
# Input:
#   - mypath: path of the ULG files
#   - outputpath: path to save the CSV files
#   - withUAVModel: boolean to indicate if the drone model is to be saved
# Output:
#   - processingTime: time in seconds that the function has taken to process the files
#####################################
def parallelUlogs2CSV(mypath, outputpath, numThread=0, filesToProcess=[], arrDesiredFiles=[]):
    summaryCSV = pd.DataFrame() # Dataframe to save the drone model of each ULG file

    # On each file in the directory
    for idx, filename in enumerate( filesToProcess ): 
        try:
            # print each X percent of the files processed
            if len(filesToProcess) > 10:
                if idx%int(len(filesToProcess)/10) == 0:
                    print(f"Thread {numThread}: Processing file {idx}/{len(filesToProcess)} ~ {idx/len(filesToProcess)*100:.2f}%")
            else:
                print(f"Thread {numThread}: Processing file {idx}/{len(filesToProcess)} ~ {idx/len(filesToProcess)*100:.2f}%")

            ulogFullPath = join(mypath, filename)
            if isfile(ulogFullPath): # OK, it's a file
                filenameNoExt = filename[:-4] # Remove the extension .ulog

                if os.stat(ulogFullPath).st_size > 0: # Check if the file is not empty
                    
                    # Read the ulog file
                    ulog = ULog(ulogFullPath, None, False)
                    #show_info(ulog, verbose=False)

                    #### Extract info on header
                    if 'sys_uuid' in ulog.msg_info_dict: # Check if dict has the key
                        sys_uuid = ulog.msg_info_dict['sys_uuid']
                    else:
                        sys_uuid = "-1"
                    if 'ver_sw' in ulog.msg_info_dict: # Check if dict has the key
                        ver_hw   = ulog.msg_info_dict['ver_hw']
                    else:
                        ver_hw = "-1"
                    if 'ver_hw_subtype' in ulog.msg_info_dict: # Check if dict has the key
                        ver_hw_subtype = ulog.msg_info_dict['ver_hw_subtype']
                    else:
                        ver_hw_subtype = "-1"
                    if 'ver_sw' in ulog.msg_info_dict: # Check if dict has the key
                        sys_name = ulog.msg_info_dict['sys_name']
                    else:
                        sys_name = "-1"
                    version  = ulog.get_version_info_str()

                    #### Extract ULOG metadata (parameters)
                    params          = ulog.initial_parameters
                    paramsSystem    = ulog.get_default_parameters(0) # Modifications after
                    paramsCurrSetup = ulog.get_default_parameters(1)
                    # Detect the drone model
                    if 'SYS_AUTOSTART' in params:   # Check if dict has the key
                        droneModel = params['SYS_AUTOSTART']
                    else:
                        droneModel = -1               # Create a new dataframe with trackID, error bool, and drone model
                    newModel = pd.DataFrame({'ULG':[filenameNoExt], 'Error': False, 'Model':[droneModel], 'SYS_NAME': sys_name, 'VER_HW': ver_hw, 'VER_FW': version})
                    
                    # Open one and extract trajectory duration, number of points and distance traveled
                    distance, duration, numPoints, bigTimeJump, bigPositionJump,           \
                    notEnoughMovementXAxis, notEnoughMovementYAxis, notEnoughMovementZAxis \
                        = extractTrajectorySummary(ulog, outputpath, filenameNoExt)
                    newModel = pd.concat([newModel, pd.DataFrame({
                        'Distance':[distance],
                        'Duration':[duration],
                        'NumPoints':[numPoints],
                        'notEnoughMovementXAxis': notEnoughMovementXAxis,
                        'notEnoughMovementYAxis': notEnoughMovementYAxis,
                        'notEnoughMovementZAxis': notEnoughMovementZAxis,
                        'bigTimeJump': bigTimeJump,
                        'bigPositionJump': bigPositionJump,
                    })], axis=1) # Add the marks of the desired files

                    ###################### SAVE RESULTS

                    #### New line in the summary CSV
                    summaryCSV = pd.concat([summaryCSV, newModel], ignore_index=True) # Concatenate the new dataframe with the previous ones

                    #### Store trajectory as CSV, if the trajectory is long enough
                    minDuration = 30; minPoints   = 30; minDistance = 30
                    # the data has to pass some checks before saving it
                    # 
                    # remove PX4_SITL on HW
                    # remove Airframe models referring to Simulation
                    if  duration      > minDuration and \
                        numPoints     > minPoints   and \
                        distance      > minDistance and \
                        ver_hw        != "PX4_SITL" and \
                        droneModel    != -1   and \
                        droneModel    != 1000 and \
                        droneModel    != 1001 and \
                        droneModel    != 1002 and \
                        droneModel    != 1100 and \
                        droneModel    != 1101 and \
                        droneModel    != 1102:
                        storeTrajectory(ulog, outputpath, filenameNoExt)

                else:
                    print(f"File {filename} is empty")
                    messageTelegramBot(f"File {filename} is empty")

        #################
        except Exception as err:
            print(f"Error en thread {numThread}, ejecutando la iteración {idx} del fichero {filename}")
            traceback.print_exception(None, err, err.__traceback__)
            messageTelegramBot(f"Error en thread {numThread}, ejecutando la iteración {idx} del fichero {filename}")

            # store this trajectory as an error
            summaryCSV = pd.concat([summaryCSV, pd.DataFrame({'ULG':[filenameNoExt], 'Error': True, 'Model':[-1]})], ignore_index=True)
            print("A")

    # Save the drone model in a CSV
    summaryCSV.to_csv(os.path.join(outputpath, f"DroneModel_{numThread}.csv"), index=False)