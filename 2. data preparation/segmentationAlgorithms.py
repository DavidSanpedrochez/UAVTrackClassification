################################
# This file includes segmentation algorithms for the UAV Track Classification project.
# 
# @author: David Sanchez <davsanch@inf.uc3m.es>
# @author: Daniel Amigo <damigo@inf.uc3m.es>
################################

#############################################################
#                       IMPORTS                             #
#############################################################
import numpy as np
import pandas as pd
import math
import matplotlib as plt

#############################################################
#                      ALGORITHMS                           #
#############################################################
# *****************************************************************************************
# ** SQUISHE Algorithm
# *******  [INPUT] DATAFRAME with the track
# *******  [INPUT] compression rate: 1/10, 1/100... 
# *******  [INPUT] minimum number of points to include in the segmented track
# *******  [OUTPUT] LIST OF INDEXES OF THE TRACK SEGMENTS
# *****************************************************************************************
def SQUISHE(track, compressionRate, minPoints,thread_id,debug=False):
    #list of indexes
    
    indexList=pd.DataFrame(data={'index': [0,1], 'distance': [0,0]})
    max_points = (compressionRate * len(track))
    if max_points<minPoints:
      max_points=minPoints
    #for each point and consecutive point in track
    for row in range(len(track)):
        if row < 2:
            continue

        new_index = pd.DataFrame(data={'index': [row], 'distance': [0]})
        indexList = pd.concat([indexList,new_index],ignore_index=True)
        #last index Li
        startPoint  = track.iloc[int(indexList.iloc[len(indexList)-3]['index'])]
        SEDPoint    = track.iloc[int(indexList.iloc[len(indexList)-2]['index'])]
        endPoint    = track.iloc[int(indexList.iloc[len(indexList)-1]['index'])]
        numerator   = SEDPoint['timestamp']-startPoint['timestamp']
        denominator = endPoint['timestamp']-startPoint['timestamp']
        if denominator==0: time_ratio = 1
        else: time_ratio = numerator/denominator

        # calculate position of SEDPoint in the line between startPoint and endPoint
        posX = startPoint['x']+(endPoint['x']-startPoint['x'])*time_ratio
        posY = startPoint['y']+(endPoint['y']-startPoint['y'])*time_ratio
        posZ = startPoint['z']+(endPoint['z']-startPoint['z'])*time_ratio

        # calculate euclidean distance 
        indexList.at[len(indexList)-2,'distance']=math.sqrt((SEDPoint['x']-posX)**2+(SEDPoint['y']-posY)**2+(SEDPoint['z']-posZ)**2)
        if debug:
            print(row)
            print(indexList)
        
        # 
        if len(indexList) > max_points:     
            #index list end
            #for each index in indexList
            ###############################################################################
            aux      = indexList.drop(len(indexList)-1)
            aux      = aux.drop(0)
            toRemove = aux['distance'].idxmin()
            SED      = indexList.iloc[toRemove]['distance']
            indexList.at[toRemove-1,'distance']=indexList.iloc[toRemove-1]['distance']+SED
            indexList.at[toRemove+1,'distance']=indexList.iloc[toRemove+1]['distance']+SED
            indexList=indexList.drop(toRemove)
            indexList=indexList.reset_index(drop=True)
            if debug:
                paintSegment(indexList,track,toRemove,row,False,True)
    
    # Return the list of indexes that define the segments
    return indexList

# *****************************************************************************************
# ** paintSegment
# *******  [INPUT] 
# *******  [OUTPUT] 
# *****************************************************************************************
def paintSegment(indexList, track, toRemove, lastTrackPoint, paintFullTrack, clearPlot):
    if paintFullTrack:
        plt.pyplot.plot(track['x'],track['y']) 
    else:
        plt.pyplot.plot(track['x'][0:lastTrackPoint],track['y'][0:lastTrackPoint])
        plt.pyplot.plot(track['x'][indexList['index']],track['y'][indexList['index']])
    #plt.gca().set_aspect("equal") 
    plt.pyplot.show()
    if clearPlot:
        plt.pyplot.clf()