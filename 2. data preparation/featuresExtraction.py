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
#############################################################
#                        FLAGS                              #
#############################################################
#############################################################
#                        PATHS                              #
#############################################################
#############################################################
#                     FUNCTIONS                             #
#############################################################
# *****************************************************************************************
# ** Calculate the increments of the track
# *******  [INPUT] DATAFRAME with the track
# *******  [INPUT] start index of the segment
# *******  [INPUT] end index of the segment
# *******  [OUTPUT] DATAFRAME with the max, min and average increments
# *****************************************************************************************
def calculateIncrements(df, start, end, thread_id, trackID, debug=False):
    # 0: maximum value 
    # 1: minimum value
    # 2: time average
    # 3: sum of all values
    features=pd.DataFrame(0,index=np.arange(4),columns=[
        'totalTime',     
        'x', 'y', 'z', 
        'vx', 'vy', 'vz',
        'ax', 'ay', 'az',
        'jx', 'jy', 'jz',
        'horDistance', 'horSpeed', 'horAcceleration', 'horJerk',
        'distance',
        'horAngle', 'verAngle', 
        'angleRate',
        'stopRate'])
    features.iloc[1]=99999
       
    #totalIncrement=pd.DataFrame()
    counter=0
    stops=0
    try:
        total_time=df.iloc[int(end)]['timestamp']-df.iloc[int(start)]['timestamp']
        prevAngleIncrement=0
        #loop from segment start to segment end
        for row in range(int(start),int(end)-1):
            incrementTime=df.iloc[row+1]['timestamp']-df.iloc[row]['timestamp']
            counter=counter+1
            #calculate current row increments
            increment_x               = abs(df.iloc[row+1]['x']-df.iloc[row]['x'])
            increment_y               = abs(df.iloc[row+1]['y']-df.iloc[row]['y'])
            increment_z               = abs(df.iloc[row+1]['z']-df.iloc[row]['z'])
            #increment_vx              = abs(df.iloc[row+1]['x']-df.iloc[row]['x'])/incrementTime
            #increment_vy              = abs(df.iloc[row+1]['y']-df.iloc[row]['y'])/incrementTime
            increment_vz              = abs(df.iloc[row+1]['z']-df.iloc[row]['z'])/incrementTime
            #increment_ax              = abs(df.iloc[row+1]['x']-df.iloc[row]['x'])/(incrementTime**2)
            #increment_ay              = abs(df.iloc[row+1]['y']-df.iloc[row]['y'])/(incrementTime**2)
            increment_az              = abs(df.iloc[row+1]['z']-df.iloc[row]['z'])/(incrementTime**2)
            #increment_jx              = abs(df.iloc[row+1]['x']-df.iloc[row]['x'])/(incrementTime**3)
            #increment_jy              = abs(df.iloc[row+1]['y']-df.iloc[row]['y'])/(incrementTime**3)
            increment_jz              = abs(df.iloc[row+1]['z']-df.iloc[row]['z'])/(incrementTime**3)
            increment_horDistance     = math.sqrt(increment_x**2+increment_y**2)
            increment_distance        = math.sqrt(increment_x**2+increment_y**2+increment_z**2)
            increment_horSpeed        = increment_horDistance     / incrementTime
            increment_horAcceleration = increment_horSpeed        / incrementTime
            increment_horJerk         = increment_horAcceleration / incrementTime
            #count horizontal stops
            if increment_horSpeed < 4: stops = stops + 1
            # angle calculation
            if row < int(end)-1:
                sumVector            = (increment_x*(abs(df.iloc[row+2]['x']-df.iloc[row+1]['x'])))+(increment_y*(abs(df.iloc[row+2]['y']-df.iloc[row+1]['y'])))
                nextHorDistIncrement = math.sqrt(abs(df.iloc[row+2]['x']-df.iloc[row+1]['x'])**2+abs(df.iloc[row+2]['y']-df.iloc[row+1]['y'])**2)
                if (increment_horDistance*nextHorDistIncrement)==0: increment_horAngle = math.acos(0)
                else:                                               increment_horAngle = math.acos(sumVector/(increment_horDistance*nextHorDistIncrement))
                prevAngleIncrement = increment_horAngle
            else: increment_horAngle = prevAngleIncrement
            increment_verAngle  = math.atan2(increment_z,increment_horDistance)
            increment_angleRate = increment_horAngle/incrementTime

            #calculate max, min and avg
            #check maximum
            if features.iloc[0]['totalTime']<incrementTime:                   features.at[0,'totalTime']=incrementTime
            #if features.iloc[0]['x']<increment_x:                             features.at[0, 'x']=increment_x
            #if features.iloc[0]['y']<increment_y:                             features.at[0,'y']=increment_y
            if features.iloc[0]['z']<increment_z:                             features.at[0,'z']=increment_z
            #if features.iloc[0]['vx']<increment_vx:                           features.at[0,'vx']=increment_vx
            #if features.iloc[0]['vy']<increment_vy:                           features.at[0,'vy']=increment_vy
            if features.iloc[0]['vz']<increment_vz:                           features.at[0,'vz']=increment_vz
            #if features.iloc[0]['ax']<increment_ax:                           features.at[0,'ax']=increment_ax
            #if features.iloc[0]['ay']<increment_ay:                           features.at[0,'ay']=increment_ay
            if features.iloc[0]['az']<increment_az:                           features.at[0,'az']=increment_az
            #if features.iloc[0]['jx']<increment_jx:                           features.at[0,'jx']=increment_jx
            #if features.iloc[0]['jy']<increment_jy:                           features.at[0,'jy']=increment_jy
            if features.iloc[0]['jz']<increment_jz:                           features.at[0,'jz']=increment_jz
            if features.iloc[0]['horDistance']<increment_horDistance:         features.at[0,'horDistance']=increment_horDistance
            if features.iloc[0]['horSpeed']<increment_horSpeed:               features.at[0,'horSpeed']=increment_horSpeed
            if features.iloc[0]['horAcceleration']<increment_horAcceleration: features.at[0,'horAcceleration']=increment_horAcceleration
            if features.iloc[0]['horJerk']<increment_horJerk:                 features.at[0,'horJerk']=increment_horJerk
            if features.iloc[0]['horAngle']<increment_horAngle:               features.at[0,'horAngle']=increment_horAngle
            if features.iloc[0]['verAngle']<increment_verAngle:               features.at[0,'verAngle']=increment_verAngle    
            if features.iloc[0]['angleRate']<increment_angleRate:             features.at[0,'angleRate']=increment_angleRate 

            #check minimum
            if features.iloc[1]['totalTime']>incrementTime:                   features.at[1,'totalTime']=incrementTime
            #if features.iloc[1]['x']>increment_x:                             features.at[1,'x']=increment_x
            #if features.iloc[1]['y']>increment_y:                             features.at[1,'y']=increment_y
            if features.iloc[1]['z']>increment_z:                             features.at[1,'z']=increment_z
            #if features.iloc[1]['vx']>increment_vx:                           features.at[1,'vx']=increment_vx
            #if features.iloc[1]['vy']>increment_vy:                           features.at[1,'vy']=increment_vy
            if features.iloc[1]['vz']>increment_vz:                           features.at[1,'vz']=increment_vz
            #if features.iloc[1]['ax']>increment_ax:                           features.at[1,'ax']=increment_ax
            #if features.iloc[1]['ay']>increment_ay:                           features.at[1,'ay']=increment_ay
            if features.iloc[1]['az']>increment_az:                           features.at[1,'az']=increment_az
            #if features.iloc[1]['jx']>increment_jx:                           features.at[1,'jx']=increment_jx
            #if features.iloc[1]['jy']>increment_jy:                           features.at[1,'jy']=increment_jy
            if features.iloc[1]['jz']>increment_jz:                           features.at[1,'jz']=increment_jz
            if features.iloc[1]['horDistance']>increment_horDistance:         features.at[1,'horDistance']=increment_horDistance
            if features.iloc[1]['horSpeed']>increment_horSpeed:               features.at[1,'horSpeed']=increment_horSpeed
            if features.iloc[1]['horAcceleration']>increment_horAcceleration: features.at[1,'horAcceleration']=increment_horAcceleration
            if features.iloc[1]['horJerk']>increment_horJerk:                 features.at[1,'horJerk']=increment_horJerk
            if features.iloc[1]['horAngle']>increment_horAngle:               features.at[1,'horAngle']=increment_horAngle
            if features.iloc[1]['verAngle']>increment_verAngle:               features.at[1,'verAngle']=increment_verAngle
            if features.iloc[1]['angleRate']>increment_angleRate:             features.at[1,'angleRate']=increment_angleRate

            # calculate sum
            #features.at[2,'x']               = increment_x+features.iloc[2]['x']
            #features.at[2,'y']               = increment_y+features.iloc[2]['y']
            features.at[2,'z']               = increment_z+features.iloc[2]['z']
            #features.at[2,'vx']              = increment_vx+features.iloc[2]['vx']
            #features.at[2,'vy']              = increment_vy+features.iloc[2]['vy']
            features.at[2,'vz']              = increment_vz+features.iloc[2]['vz']
            #features.at[2,'ax']              = increment_ax+features.iloc[2]['ax']
            #features.at[2,'ay']              = increment_ay+features.iloc[2]['ay']
            features.at[2,'az']              = increment_az+features.iloc[2]['az']
            #features.at[2,'jx']              = increment_jx+features.iloc[2]['jx']
            #features.at[2,'jy']              = increment_jy+features.iloc[2]['jy']
            features.at[2,'jz']              = increment_jz+features.iloc[2]['jz']
            features.at[2,'horDistance']     = increment_horDistance+features.iloc[2]['horDistance']
            features.at[2,'horSpeed']        = increment_horSpeed+features.iloc[2]['horSpeed']
            features.at[2,'horAcceleration'] = increment_horAcceleration+features.iloc[2]['horAcceleration']
            features.at[2,'horJerk']         = increment_horJerk+features.iloc[2]['horJerk']
            features.at[2,'horAngle']        = increment_horAngle+features.iloc[2]['horAngle']
            features.at[2,'verAngle']        = increment_verAngle+features.iloc[2]['verAngle']
            features.at[2,'angleRate']       = increment_angleRate+features.iloc[2]['angleRate']

            # sum of the increments
            features.at[3,'horDistance'] += increment_horDistance
            features.at[3,'z']           += increment_z
            features.at[3,'distance']    += increment_distance
            if counter==0: features.at[3,'stopRate'] = 0
            else:          features.at[3,'stopRate'] = stops/counter

        ###### LOOP END ######

        #calculate rates over time
        #features.at[2,'x']               = features.iloc[2]['x'] / total_time
        #features.at[2,'y']               = features.iloc[2]['y'] / total_time
        features.at[2,'z']               = features.iloc[2]['z'] / total_time
        #features.at[2,'vx']              = features.iloc[2]['vx'] / total_time
        #features.at[2,'vy']              = features.iloc[2]['vy'] / total_time
        features.at[2,'vz']              = features.iloc[2]['vz'] / total_time
        #features.at[2,'ax']              = features.iloc[2]['ax'] / total_time
        #features.at[2,'ay']              = features.iloc[2]['ay'] / total_time
        features.at[2,'az']              = features.iloc[2]['az'] / total_time
        #features.at[2,'jx']              = features.iloc[2]['jx'] / total_time
        #features.at[2,'jy']              = features.iloc[2]['jy'] / total_time
        features.at[2,'jz']              = features.iloc[2]['jz'] / total_time
        features.at[2,'horDistance']     = features.iloc[2]['horDistance'] / total_time
        features.at[2,'horSpeed']        = features.iloc[2]['horSpeed'] / total_time
        features.at[2,'horAcceleration'] = features.iloc[2]['horAcceleration'] / total_time
        features.at[2,'horJerk']         = features.iloc[2]['horJerk'] / total_time
        features.at[2,'horAngle']        = features.iloc[2]['horAngle'] / total_time
        features.at[2,'verAngle']        = features.iloc[2]['verAngle'] / total_time
        features.at[2,'angleRate']       = features.iloc[2]['angleRate'] / total_time

        # sum of all values
        features.at[3,'totalTime']        = total_time

        if debug:
            print(features) 

    except Exception as err2:
        print(f"Error en thread {thread_id}, ejecutando el calculo de incrementos en track: {trackID}")
        print(err2)  
    return features


# *****************************************************************************************
# feature extraction from the tracks
#GOES THROUGH EVERY FILE AND DIRECTORY
#TRANSFORMS DATA TO INTERVALS
#SAVES EVERYTHING IN A CSV (output.csv)

# *******  [INPUT] tracksDir:  path to the folder with the tracks
# *******  [INPUT] outputPath: path to the folder to store the procesed features
# *******  [INPUT] :           flag to indicate if segmentation can be applied
# *******  [INPUT] thread_id:  thread  number
# *******  [INPUT] tracks:     list of tracks to be processed 
# *******  [INPUT] models:     drone model for each track
# *******  [INPUT] segments:   list with the calculated segments
# *******  [OUTPUT] CSV file with the dataframe including the features extracted from the tracks
# *****************************************************************************************
# DETAILED LIST OF FEATURES
# Based on previous work of ship classification:
# A1. Segment/Track {total} time 
# A2. Segment/Track {total/max/min} increment ponderated with the time in position in coordinates [x,y,z] 
# A3. Segment/Track {total/max/min} increment ponderated with the time in velocity (postion increment/total time) in coordinates [x,y,z]
# A4. Segment/Track {total/max/min} increment ponderated with the time in acceleration (velocity increment/total time) in coordinates [x,y,z]
#
# Additional features:
# B1. Segment/Track {total/max/min} increment ponderated with the time in jerk (acceleration increment/total time) in coordinates [x,y,z]
# B2. Segment/Track {total/max/min} Horizontal plane (xy) vector modulus (distance,velocity,acceleration,jerk)
#
# Based on references on UAV classification:
# C1. Segment/Track {total/max/min} Heading increment / Horizontal angle (angle between two consecutive segments) [https://ieeexplore.ieee.org/document/9925778 https://www.sto.nato.int/publications/STO%20Meeting%20Proceedings/STO-MP-MSG-SET-183/MP-MSG-SET-183-06P.pdf] - Taking into account 0=360
# C2. Segment/Track {total/max/min} Slope angle (vertical angle increment ) [https://ieeexplore.ieee.org/document/9925778 https://ieeexplore.ieee.org/document/6875676] - Range is [-pi/2,pi/2)
# C3. Segment/Track {total/max/min} Turn rate (horizontal angle increment divided by time) [https://ieeexplore.ieee.org/document/9011293]
# C4. Segment/Track {total} Stop rate  (rate of consecutive points under 4 m/s) [https://www.sto.nato.int/publications/STO%20Meeting%20Proceedings/STO-MP-MSG-SET-183/MP-MSG-SET-183-06P.pdf]
#
# Additional attributes:
# UAV model (class to be processed)
# Track identifier (for data management and results understanding)
# Segment identifier (for data management and results understanding)
# # *****************************************************************************************
def featuresExtraction(tracksDir, outputPath, thread_id, tracks, models, segments, debug=False):
    
    tracks = tracks.reset_index(drop=True)
    models = models.reset_index(drop=True)
    if debug:
        print("thread_id")
        print(thread_id)

    df_featuresList = pd.DataFrame(
        columns=[                       # See detailed features list above
            ############################# SEGMENT FEATURES #############################
            ## TOTAL
            'totalTime_Segment',#A1
            'totalHorDistance_Segment',
            'totalVerDistance_Segment',
            'totalDistance_Segment',
            'stopRate_Segment',#C4
            ###### XY
            'max_horSpeed_Segment', 'max_horAcceleration_Segment', 'max_horJerk_Segment',#B2
            'min_horSpeed_Segment', 'min_horAcceleration_Segment', 'min_horJerk_Segment',#B2
            'timeRatedTotal_horDistance_Segment', 'timeRatedTotal_horSpeed_Segment', 'timeRatedTotal_horAcceleration_Segment', 'timeRatedTotal_horJerk_Segment',#B2
            #### Z AXIS
            'max_z_Segment', #A2 
            'min_z_Segment', #A2 
            'timeRatedTotal_z_Segment', #A2 
            'max_vz_Segment',#A3 
            'min_vz_Segment',#A3
            'timeRatedTotal_vz_Segment',#A3
            'max_az_Segment',#A4
            'min_az_Segment',#A4
            'timeRatedTotal_az_Segment',#A4
            'max_jz_Segment',#B1
            'min_jz_Segment',#B1
            'timeRatedTotal_jz_Segment',#B1
            #### ANGLES
            'max_horAngle_Segment', 'max_verAngle_Segment',#C1-C2 
            'min_horAngle_Segment', 'min_verAngle_Segment', #C1-C2 
            'timeRatedTotal_horAngle_Segment', 'timeRatedTotal_verAngle_Segment', #C1-C2 
            'max_angleRate_Segment',#C3
            'min_angleRate_Segment',#C3
            'timeRatedTotal_angleRate_Segment',#C3
            ############### FULL TRACK ################
            ## TOTAL
            'totalTime_Track',#A1
            'totalHorDistance_Track',
            'totalVerDistance_Track',
            'totalDistance_Track',
            'stopRate_Track',#C4
            ### XY
            'max_horSpeed_Track', 'max_horAcceleration_Track', 'max_horJerk_Track',#B2
            'min_horSpeed_Track', 'min_horAcceleration_Track', 'min_horJerk_Track',#B2
            'timeRatedTotal_horDistance_Track', 'timeRatedTotal_horSpeed_Track', 'timeRatedTotal_horAcceleration_Track', 'timeRatedTotal_horJerk_Track',#B2
            #### Z AXIS	
            'max_z_Track',#A2 
            'min_z_Track',#A2 
            'timeRatedTotal_z_Track', #A2
            'max_vz_Track',#A3 
            'min_vz_Track',#A3 
            'timeRatedTotal_vz_Track',#A3 
            'max_az_Track',#A4
            'min_az_Track',#A4
            'timeRatedTotal_az_Track',#A4
            'max_jz_Track',#B1
            'min_jz_Track',#B1
            'timeRatedTotal_jz_Track',#B1
            #### ANGLES
            'max_horAngle_Track', 'max_verAngle_Track',#C1-C2 
            'min_horAngle_Track', 'min_verAngle_Track', #C1-C2
            'timeRatedTotal_horAngle_Track', 'timeRatedTotal_verAngle_Track', #C1-C2
            'max_angleRate_Track',#C3
            'min_angleRate_Track',#C3
            'timeRatedTotal_angleRate_Track',#C3
            ##### 
            'UAV_Airframe','seg_id_track','seg_id_unique','track_id'
        ] #Extra attributes 
    )

    # 
    if debug: print(tracks)

    # for each track in x_train and model in y_train
    for i in range(len(tracks)): # for each track
        try:
            currentDroneModel = models[i] # Get drone model name
            trackFile         = tracks[i]+".csv"
            # Find segments of this track
            segmentsThisTrack = segments[['track_id']].apply(lambda x: x.str.contains(tracks[i],case=False)).any(axis=1).astype(int)
            segmentsThisTrack = segments[segmentsThisTrack==1]
        except:
            print("Error: "+str(i)+" "+str(len(tracks))+" "+str(len(models)))

        # Print the percentage of tracks processed each 10 tracks
        if i%10==0:
            print("Thread "+str(thread_id)+" has processed "+str(i)+" tracks of "+str(len(tracks))+" ("+str(round(i/len(tracks)*100,2))+"%)")

        # Read track file
        trackPathFile = join(tracksDir, trackFile)
        try:
            if isfile(trackPathFile):
                df = pd.read_csv(trackPathFile,sep=',')
                trackIncrements = calculateIncrements(df, 0, len(df)-1, thread_id, tracks[i])

                #apply SQUISHE algorithm
                new_data = pd.DataFrame(columns=[                       # See detailed features list above
                    ############################# SEGMENT FEATURES #############################
                    ## TOTAL
                    'totalTime_Segment',#A1
                    'totalHorDistance_Segment',
                    'totalVerDistance_Segment',
                    'totalDistance_Segment',
                    'stopRate_Segment',#C4
                    ###### XY
                    'max_horSpeed_Segment', 'max_horAcceleration_Segment', 'max_horJerk_Segment',#B2
                    'min_horSpeed_Segment', 'min_horAcceleration_Segment', 'min_horJerk_Segment',#B2
                    'timeRatedTotal_horDistance_Segment', 'timeRatedTotal_horSpeed_Segment', 'timeRatedTotal_horAcceleration_Segment', 'timeRatedTotal_horJerk_Segment',#B2
                    #### Z AXIS
                    'max_z_Segment',#A2 
                    'min_z_Segment', #A2 
                    'timeRatedTotal_z_Segment', #A2 
                    'max_vz_Segment',#A3 
                    'min_vz_Segment',#A3
                    'timeRatedTotal_vz_Segment',#A3
                    'max_az_Segment',#A4
                    'min_az_Segment',#A4
                    'timeRatedTotal_az_Segment',#A4
                    'max_jz_Segment',#B1
                    'min_jz_Segment',#B1
                    'timeRatedTotal_jz_Segment',#B1
                    #### ANGLES
                    'max_horAngle_Segment', 'max_verAngle_Segment',#C1-C2 
                    'min_horAngle_Segment', 'min_verAngle_Segment', #C1-C2 
                    'timeRatedTotal_horAngle_Segment', 'timeRatedTotal_verAngle_Segment', #C1-C2 
                    'max_angleRate_Segment',#C3
                    'min_angleRate_Segment',#C3
                    'timeRatedTotal_angleRate_Segment',#C3
                    ############### FULL TRACK ################
                    ## TOTAL
                    'totalTime_Track',#A1
                    'totalHorDistance_Track',
                    'totalVerDistance_Track',
                    'totalDistance_Track',
                    'stopRate_Track',#C4
                    ### XY
                    'max_horSpeed_Track', 'max_horAcceleration_Track', 'max_horJerk_Track',#B2
                    'min_horSpeed_Track', 'min_horAcceleration_Track', 'min_horJerk_Track',#B2
                    'timeRatedTotal_horDistance_Track', 'timeRatedTotal_horSpeed_Track', 'timeRatedTotal_horAcceleration_Track', 'timeRatedTotal_horJerk_Track',#B2
                    #### Z AXIS	
                    'max_z_Track',#A2 
                    'min_z_Track',#A2 
                    'timeRatedTotal_z_Track', #A2
                    'max_vz_Track',#A3 
                    'min_vz_Track',#A3 
                    'timeRatedTotal_vz_Track',#A3 
                    'max_az_Track',#A4
                    'min_az_Track',#A4
                    'timeRatedTotal_az_Track',#A4
                    'max_jz_Track',#B1
                    'min_jz_Track',#B1
                    'timeRatedTotal_jz_Track',#B1
                    #### ANGLES
                    'max_horAngle_Track', 'max_verAngle_Track',#C1-C2 
                    'min_horAngle_Track', 'min_verAngle_Track', #C1-C2
                    'timeRatedTotal_horAngle_Track', 'timeRatedTotal_verAngle_Track', #C1-C2
                    'max_angleRate_Track',#C3
                    'min_angleRate_Track',#C3
                    'timeRatedTotal_angleRate_Track',#C3
                    ##### 
                    'UAV_Airframe','seg_id_track','seg_id_unique','track_id'] #Extra attributes 
                )

                # Read segments from track
                ################### AAAAAAAAAAAAAAAAAAA
                segmentID=0
                for j in range(len(segmentsThisTrack)-1): #for each segment of this track
                    segmentStart = int(segmentsThisTrack.iloc[j].segmentStart)
                    segmentEnd   = int(segmentsThisTrack.iloc[j].segmentEnd)
                    # for each segment of the track, calculate the features
                    segmentIncrement = calculateIncrements(df, segmentStart, segmentEnd, thread_id, tracks[i])
                    #concatenate all data
                    # Segment related features - totals
                    new_data.at[0,'totalTime_Segment']        = segmentIncrement.iloc[3]['totalTime']    
                    new_data.at[0,'totalHorDistance_Segment'] = segmentIncrement.iloc[3]['horDistance']    
                    new_data.at[0,'totalVerDistance_Segment'] = segmentIncrement.iloc[3]['z']    
                    new_data.at[0,'totalDistance_Segment']    = segmentIncrement.iloc[3]['distance']    
                    new_data.at[0,'stopRate_Segment']         = segmentIncrement.iloc[3]['stopRate']
                    
                    # Segment related features - maximums, minimums and time rated totals
                    #new_data.at[0,'max_x_Segment']                           = segmentIncrement.iloc[0]['x']
                    #new_data.at[0,'max_y_Segment']                           = segmentIncrement.iloc[0]['y']
                    new_data.at[0,'max_z_Segment']                           = segmentIncrement.iloc[0]['z']
                    #new_data.at[0,'min_x_Segment']                           = segmentIncrement.iloc[1]['x']
                    #new_data.at[0,'min_y_Segment']                           = segmentIncrement.iloc[1]['y']
                    new_data.at[0,'min_z_Segment']                           = segmentIncrement.iloc[1]['z']
                    #new_data.at[0,'timeRatedTotal_x_Segment']                = segmentIncrement.iloc[2]['x']
                    #new_data.at[0,'timeRatedTotal_y_Segment']                = segmentIncrement.iloc[2]['y']
                    new_data.at[0,'timeRatedTotal_z_Segment']                = segmentIncrement.iloc[2]['z']
                    #new_data.at[0,'max_vx_Segment']                          = segmentIncrement.iloc[0]['vx']
                    #new_data.at[0,'max_vy_Segment']                          = segmentIncrement.iloc[0]['vy']
                    new_data.at[0,'max_vz_Segment']                          = segmentIncrement.iloc[0]['vz']
                    #new_data.at[0,'min_vx_Segment']                          = segmentIncrement.iloc[1]['vx']
                    #new_data.at[0,'min_vy_Segment']                          = segmentIncrement.iloc[1]['vy']
                    new_data.at[0,'min_vz_Segment']                          = segmentIncrement.iloc[1]['vz']
                    #new_data.at[0,'timeRatedTotal_vx_Segment']               = segmentIncrement.iloc[2]['vx']
                    #new_data.at[0,'timeRatedTotal_vy_Segment']               = segmentIncrement.iloc[2]['vy']
                    new_data.at[0,'timeRatedTotal_vz_Segment']               = segmentIncrement.iloc[2]['vz']
                    #new_data.at[0,'max_ax_Segment']                          = segmentIncrement.iloc[0]['ax']
                    #new_data.at[0,'max_ay_Segment']                          = segmentIncrement.iloc[0]['ay']
                    new_data.at[0,'max_az_Segment']                          = segmentIncrement.iloc[0]['az']
                    #new_data.at[0,'min_ax_Segment']                          = segmentIncrement.iloc[1]['ax']
                    #new_data.at[0,'min_ay_Segment']                          = segmentIncrement.iloc[1]['ay']
                    new_data.at[0,'min_az_Segment']                          = segmentIncrement.iloc[1]['az']
                    #new_data.at[0,'timeRatedTotal_ax_Segment']               = segmentIncrement.iloc[2]['ax']
                    #new_data.at[0,'timeRatedTotal_ay_Segment']               = segmentIncrement.iloc[2]['ay']
                    new_data.at[0,'timeRatedTotal_az_Segment']               = segmentIncrement.iloc[2]['az']
                    #new_data.at[0,'max_jx_Segment']                          = segmentIncrement.iloc[0]['jx']
                    #new_data.at[0,'max_jy_Segment']                          = segmentIncrement.iloc[0]['jy']
                    new_data.at[0,'max_jz_Segment']                          = segmentIncrement.iloc[0]['jz']
                    #new_data.at[0,'min_jx_Segment']                          = segmentIncrement.iloc[1]['jx']
                    #new_data.at[0,'min_jy_Segment']                          = segmentIncrement.iloc[1]['jy']
                    new_data.at[0,'min_jz_Segment']                          = segmentIncrement.iloc[1]['jz']
                    #new_data.at[0,'timeRatedTotal_jx_Segment']               = segmentIncrement.iloc[2]['jx']
                    #new_data.at[0,'timeRatedTotal_jy_Segment']               = segmentIncrement.iloc[2]['jy']
                    new_data.at[0,'timeRatedTotal_jz_Segment']               = segmentIncrement.iloc[2]['jz']
                    #new_data.at[0,'max_horDistance_Segment']                 = segmentIncrement.iloc[0]['horDistance']
                    #new_data.at[0,'min_horDistance_Segment']                 = segmentIncrement.iloc[1]['horDistance']
                    new_data.at[0,'timeRatedTotal_horDistance_Segment']      = segmentIncrement.iloc[2]['horDistance']
                    new_data.at[0,'max_horSpeed_Segment']                    = segmentIncrement.iloc[0]['horSpeed']
                    new_data.at[0,'min_horSpeed_Segment']                    = segmentIncrement.iloc[1]['horSpeed']
                    new_data.at[0,'timeRatedTotal_horSpeed_Segment']         = segmentIncrement.iloc[2]['horSpeed']
                    new_data.at[0,'max_horAcceleration_Segment']             = segmentIncrement.iloc[0]['horAcceleration']
                    new_data.at[0,'min_horAcceleration_Segment']             = segmentIncrement.iloc[1]['horAcceleration']
                    new_data.at[0,'timeRatedTotal_horAcceleration_Segment']  = segmentIncrement.iloc[2]['horAcceleration']
                    new_data.at[0,'max_horJerk_Segment']                     = segmentIncrement.iloc[0]['horJerk']
                    new_data.at[0,'min_horJerk_Segment']                     = segmentIncrement.iloc[1]['horJerk']
                    new_data.at[0,'timeRatedTotal_horJerk_Segment']          = segmentIncrement.iloc[2]['horJerk']
                    new_data.at[0,'max_horAngle_Segment']                    = segmentIncrement.iloc[0]['horAngle']
                    new_data.at[0,'min_horAngle_Segment']                    = segmentIncrement.iloc[1]['horAngle']
                    new_data.at[0,'timeRatedTotal_horAngle_Segment']         = segmentIncrement.iloc[2]['horAngle']
                    new_data.at[0,'max_verAngle_Segment']                    = segmentIncrement.iloc[0]['verAngle']
                    new_data.at[0,'min_verAngle_Segment']                    = segmentIncrement.iloc[1]['verAngle']
                    new_data.at[0,'timeRatedTotal_verAngle_Segment']         = segmentIncrement.iloc[2]['verAngle']
                    new_data.at[0,'max_angleRate_Segment']                   = segmentIncrement.iloc[0]['angleRate']
                    new_data.at[0,'min_angleRate_Segment']                   = segmentIncrement.iloc[1]['angleRate']
                    new_data.at[0,'timeRatedTotal_angleRate_Segment']        = segmentIncrement.iloc[2]['angleRate']
                    ################################### FULL TRACK
                    # Track related features - totals
                    new_data.at[0,'totalTime_Track']        = trackIncrements.iloc[3]['totalTime']    
                    new_data.at[0,'totalHorDistance_Track'] = trackIncrements.iloc[3]['horDistance']    
                    new_data.at[0,'totalVerDistance_Track'] = trackIncrements.iloc[3]['z']    
                    new_data.at[0,'totalDistance_Track']    = trackIncrements.iloc[3]['distance']    
                    new_data.at[0,'stopRate_Track']         = trackIncrements.iloc[3]['stopRate']
                    # Track related features - max, min, timeRatedTotal
                    #new_data.at[0,'max_x_Track']                             = trackIncrements.iloc[0]['x']
                    #new_data.at[0,'max_y_Track']                             = trackIncrements.iloc[0]['y']
                    new_data.at[0,'max_z_Track']                             = trackIncrements.iloc[0]['z']
                    #new_data.at[0,'min_x_Track']                             = trackIncrements.iloc[1]['x']
                    #new_data.at[0,'min_y_Track']                             = trackIncrements.iloc[1]['y']
                    new_data.at[0,'min_z_Track']                             = trackIncrements.iloc[1]['z']
                    #new_data.at[0,'timeRatedTotal_x_Track']                  = trackIncrements.iloc[2]['x']
                    #new_data.at[0,'timeRatedTotal_y_Track']                  = trackIncrements.iloc[2]['y']
                    new_data.at[0,'timeRatedTotal_z_Track']                  = trackIncrements.iloc[2]['z']
                    #new_data.at[0,'max_vx_Track']                            = trackIncrements.iloc[0]['vx']
                    #new_data.at[0,'max_vy_Track']                            = trackIncrements.iloc[0]['vy']
                    new_data.at[0,'max_vz_Track']                            = trackIncrements.iloc[0]['vz']
                    #new_data.at[0,'min_vx_Track']                            = trackIncrements.iloc[1]['vx']
                    #new_data.at[0,'min_vy_Track']                            = trackIncrements.iloc[1]['vy']
                    new_data.at[0,'min_vz_Track']                            = trackIncrements.iloc[1]['vz']
                    #new_data.at[0,'timeRatedTotal_vx_Track']                 = trackIncrements.iloc[2]['vx']
                    #new_data.at[0,'timeRatedTotal_vy_Track']                 = trackIncrements.iloc[2]['vy']
                    new_data.at[0,'timeRatedTotal_vz_Track']                 = trackIncrements.iloc[2]['vz']
                    #new_data.at[0,'max_ax_Track']                            = trackIncrements.iloc[0]['ax']
                    #new_data.at[0,'max_ay_Track']                            = trackIncrements.iloc[0]['ay']
                    new_data.at[0,'max_az_Track']                            = trackIncrements.iloc[0]['az']
                    #new_data.at[0,'min_ax_Track']                            = trackIncrements.iloc[1]['ax']
                    #new_data.at[0,'min_ay_Track']                            = trackIncrements.iloc[1]['ay']
                    new_data.at[0,'min_az_Track']                            = trackIncrements.iloc[1]['az']
                    #new_data.at[0,'timeRatedTotal_ax_Track']                 = trackIncrements.iloc[2]['ax']
                    #new_data.at[0,'timeRatedTotal_ay_Track']                 = trackIncrements.iloc[2]['ay']
                    new_data.at[0,'timeRatedTotal_az_Track']                 = trackIncrements.iloc[2]['az']
                    #new_data.at[0,'max_jx_Track']                            = trackIncrements.iloc[0]['jx']
                    #new_data.at[0,'max_jy_Track']                            = trackIncrements.iloc[0]['jy']
                    new_data.at[0,'max_jz_Track']                            = trackIncrements.iloc[0]['jz']
                    #new_data.at[0,'min_jx_Track']                            = trackIncrements.iloc[1]['jx']
                    #new_data.at[0,'min_jy_Track']                            = trackIncrements.iloc[1]['jy']
                    new_data.at[0,'min_jz_Track']                            = trackIncrements.iloc[1]['jz']
                    #new_data.at[0,'timeRatedTotal_jx_Track']                 = trackIncrements.iloc[2]['jx']
                    #new_data.at[0,'timeRatedTotal_jy_Track']                 = trackIncrements.iloc[2]['jy']
                    new_data.at[0,'timeRatedTotal_jz_Track']                 = trackIncrements.iloc[2]['jz']
                    #new_data.at[0,'max_horDistance_Track']                   = trackIncrements.iloc[0]['horDistance']
                    #new_data.at[0,'min_horDistance_Track']                   = trackIncrements.iloc[1]['horDistance']
                    new_data.at[0,'timeRatedTotal_horDistance_Track']        = trackIncrements.iloc[2]['horDistance']
                    new_data.at[0,'max_horSpeed_Track']                      = trackIncrements.iloc[0]['horSpeed']
                    new_data.at[0,'min_horSpeed_Track']                      = trackIncrements.iloc[1]['horSpeed']
                    new_data.at[0,'timeRatedTotal_horSpeed_Track']           = trackIncrements.iloc[2]['horSpeed']
                    new_data.at[0,'max_horAcceleration_Track']               = trackIncrements.iloc[0]['horAcceleration']
                    new_data.at[0,'min_horAcceleration_Track']               = trackIncrements.iloc[1]['horAcceleration']
                    new_data.at[0,'timeRatedTotal_horAcceleration_Track']    = trackIncrements.iloc[2]['horAcceleration']
                    new_data.at[0,'max_horJerk_Track']                       = trackIncrements.iloc[0]['horJerk']
                    new_data.at[0,'min_horJerk_Track']                       = trackIncrements.iloc[1]['horJerk']
                    new_data.at[0,'timeRatedTotal_horJerk_Track']            = trackIncrements.iloc[2]['horJerk']
                    new_data.at[0,'max_horAngle_Track']                      = trackIncrements.iloc[0]['horAngle']
                    new_data.at[0,'min_horAngle_Track']                      = trackIncrements.iloc[1]['horAngle']
                    new_data.at[0,'timeRatedTotal_horAngle_Track']           = trackIncrements.iloc[2]['horAngle']
                    new_data.at[0,'max_verAngle_Track']                      = trackIncrements.iloc[0]['verAngle']
                    new_data.at[0,'min_verAngle_Track']                      = trackIncrements.iloc[1]['verAngle']
                    new_data.at[0,'timeRatedTotal_verAngle_Track']           = trackIncrements.iloc[2]['verAngle']
                    new_data.at[0,'max_angleRate_Track']                     = trackIncrements.iloc[0]['angleRate']
                    new_data.at[0,'min_angleRate_Track']                     = trackIncrements.iloc[1]['angleRate']
                    new_data.at[0,'timeRatedTotal_angleRate_Track']          = trackIncrements.iloc[2]['angleRate']
                    # Additional attributes
                    new_data.at[0,'UAV_Airframe']  = currentDroneModel
                    new_data.at[0,'seg_id_track']  = segments['seg_id'][j]
                    new_data.at[0,'seg_id_unique'] = segments['seg_id_unique'][j]
                    new_data.at[0,'track_id']      = tracks[i]
                    
                    # next segment in this track
                    segmentID = segmentID+1
                    # Concatenate the new segment with the rest of the segments of this thread
                    df_featuresList = pd.concat([df_featuresList, new_data], ignore_index=True)
                    # Save the new segment on the file of this thread (overwrite all the time)
                    df_featuresList.to_csv(os.path.join(outputPath, f"features_{thread_id}.csv"), index=False)

        # If any error occurs during the feature extraction of this track, it is skipped
        except Exception as err:
            print(f"Error en thread {thread_id}, ejecutando la iteración {i} del fichero {trackFile}")
            print(err)  

    # At the end of all the tracks, save the features of this thread on the file
    df_featuresList.to_csv(os.path.join(outputPath, f"features_{thread_id}.csv"), index=False)