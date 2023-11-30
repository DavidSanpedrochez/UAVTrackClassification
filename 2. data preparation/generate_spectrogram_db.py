################################
# This script contains the spectrogram generation functions
# 
# @author: David Sanchez <davsanch@inf.uc3m.es>
# @author: Daniel Amigo <damigo@inf.uc3m.es>
# @author: Paco Fariña  <franciscofarinasalguero@gmail.com>
################################
#############################################################
#                       IMPORTS                             #
############################################################# 
import torch
import numpy as np 
import  pywt

import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from  utils import _calculateMagnitude,_calculateAcceleration, _movement_flight_features, _filter_by_duration,_plot_spectrogram
from  utils import _flight_spectrogram
import os 
import os.path
import shutil

from tqdm import tqdm

#############################################################
#                        FLAGS                              #
#############################################################
#############################################################
#                        PATHS                              #
#############################################################
DATA_PATH= "C:/Users/pacofarina/Documents/TFM/dataset/finalisimov2/" # folder containing each flight as a <ulg>.csv
METADATA_PATH= "C:/Users/pacofarina/Documents/TFM/dataset/Dataset final.csv"# file containg metadata for each flight (including ulg)
OUTPUT_PATH="C:/Users/pacofarina/davidUAVClassification-2/Data/Spectrogram Datasets/Overlapping"
SPECTROGRAM_DB_PATH="Data/Spectrogram Datasets/Thresholds/5000"

#############################################################
#                     FUNCTIONS                             #
#############################################################

# *****************************************************************************************
# ** Generate an spectrogram with overlapp
# *******  [INPUT] size: segment size in number of waypoints
# *******  [INPUT] plot: if True, save a figure that plots the generated spectrogram
# *******  [INPUT] overlap: number of waypoints of overlap between segments
# ******* [OUTPUT] filtered_df: dataframe with the defined spectrograms
# ******* [OUTPUT] outputPath: path with the generated plots of spectrograms
# *****************************************************************************************
def overlapping(size=300, plot=False, overlap=10, outputPath=OUTPUT_PATH):

    # Define output dataframes
    overlapping_df= pd.DataFrame(columns=["spectrogram", "label","ULG"])
    # Generate a dataframe with metadata for each trajectory 
    metadata= pd.read_csv(METADATA_PATH)
    # Select the trajectories with enough points for segmentation
    filtered_df= metadata[metadata["NumPoints"]>size]
    filtered_df["spectrogram"]=0

    # Progress bar for visualization
    total_rows = len(filtered_df)
    pbar = tqdm(total=total_rows)
    # Count with the number of trajectories of Quadrotor x
    quadx_count=0
    
    # Iterate over selected trajectories
    for index, row in filtered_df.iterrows():
        # Get the trajectory metadata 
        filename = row["ULG"]
        label = row["Class"]
        ulg=row["ULG"]

        # if os.path.exists("Data/Spectrogram Datasets/Overlapping/size"+str(size)+"_overlap10/"+label)==False:
        #     os.mkdir("Data/Spectrogram Datasets/Overlapping/size"+str(size)+"_overlap10/"+ label)

        # Load trajectory waypoints
        full_flight= pd.read_csv(DATA_PATH+ filename+".csv")
        # Split the trajectory into segments of defined size, and with 10 measurements of overlap on each side 
        split_dfs = [full_flight.loc[i:i+size-1,:] for i in range(0, len(full_flight),size-overlap) if i <len(full_flight)-overlap]

        # If the last segment is smaller than size, pad it with zeros to reach desired size
        while len(split_dfs[-1])<size:
            df2 = pd.DataFrame([[0]*split_dfs[-1].shape[1]],columns=split_dfs[-1].columns)
            split_dfs[-1] = split_dfs[-1].append(df2, ignore_index=True)
            #del df2  

        # Count the number of trajectories of Quadrotor x   
        if label=="Quadrotor x":
            quadx_count+=len(split_dfs)

        # Only 20000 trajectories of Quadrotor x are used, the rest are discarded. 
        # Other classes are used without limitation
        if (label!="Quadrotor x") | (quadx_count<20000):
            for flight_segment, segment_id in zip(split_dfs, range(0,len(split_dfs)-1)):
                # Extract features from generated segments           
                flight= _movement_flight_features(flight_segment)
                # Calculate the spectrogram for each segment  
                D= _flight_spectrogram(flight)
                # Plot the spectrogram if plot=True
                if plot==True:
                    _plot_spectrogram(D, filepath= outputPath+"/size"+str(size)+"_overlap"+str(overlap)+"/"+label+"/"+filename+"_"+str(segment_id)+label)
                
                #overlapping_df=overlapping_df.append({"spectrogram": D, "label": label, "ULG":ulg},ignore_index=True)
                ## Free up memory
                #del flight, D            
            #free memory
            #del filename,label,ulg
        #Update progress bar    
        pbar.update(1)
    # Generate output csv
    overlapping_df.to_csv(outputPath+"/size"+str(size)+"_overlap"+str(overlap)+"/spectrograms.csv")
    pbar.close()
    # Return the dataframe with spectrograms and labels
    return filtered_df
# *****************************************************************************************
# ** Combine subclasses into a single class
# *****************************************************************************************
def _combine_sub_classes():

    # Define the source and destination directories
    src_dir = "C:\\Users\\pacofarina\\davidUAVClassification-2\\Data\\Spectrogram Datasets\\Overlapping\\size300_overlap10"

    dst_dir = os.path.join(src_dir, "combined")
    
    # Define the classes to be combined
    classes = {
        "Hexarotor": ["Hexarotor +", "Hexarotor Coaxial", "Hexarotor x"],
        "Octorotor": ["Octorotor Coaxial", "Octorotor x"],
        "Plane": ["Plane A-Tail", "Plane V-Tail"],
        "Quadrotor": ["Quadrotor Wide", "Quadrotor x"],
        "VTOL": ["VTOL"]
    }

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # Loop through the classes
    for class_name, class_dirs in classes.items():
        # Create the class directory in the destination
        class_dir = os.path.join(dst_dir, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        # Loop through the source directories for the class
        for class_dir_src in class_dirs:
            # Combine the source and destination paths
            class_dir_src = os.path.join(src_dir, class_dir_src)
            # Loop through the files in the source directory
            for filename in os.listdir(class_dir_src):
                # Combine the source and destination paths
                src_path = os.path.join(class_dir_src, filename)
                dst_path = os.path.join(class_dir, filename)
                # Copy the file to the destination
                shutil.copy(src_path, dst_path)


#############################################################
#                        MAIN                               #
#############################################################
def main():
    
    
    #overlapping(size=300, plot=True)
    _combine_sub_classes()


if __name__ == "__main__":
    main()










