################################
# This script contains the spectrogram generation functions
# 
# @author: David Sanchez <davsanch@inf.uc3m.es>
# @author: Daniel Amigo <damigo@inf.uc3m.es>
# @author: Paco Fariña  <franciscofarinasalguero@gmail.com>
################################
#imports 
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






#Datapaths
DATA_PATH= "C:/Users/pacofarina/Documents/TFM/dataset/finalisimov2/" # folder containing each flight as a <ulg>.csv
METADATA_PATH= "C:/Users/pacofarina/Documents/TFM/dataset/Dataset final.csv"# file containg metadata for each flight (including ulg)

SPECTROGRAM_DB_PATH="Data/Spectrogram Datasets/Thresholds/5000"


def overlapping(size=300, plot=False):

    overlapping_df= pd.DataFrame(columns=["spectrogram", "label","ULG"])
    metadata= pd.read_csv(METADATA_PATH)
    filtered_df= metadata[metadata["NumPoints"]>size]
    filtered_df["spectrogram"]=0

    total_rows = len(filtered_df)
    pbar = tqdm(total=total_rows)
    quadx_count=0
    



    for index, row in filtered_df.iterrows():
        filename = row["ULG"]
        label = row["Class"]
        ulg=row["ULG"]

        # if os.path.exists("Data/Spectrogram Datasets/Overlapping/size"+str(size)+"_overlap10/"+label)==False:
        #     os.mkdir("Data/Spectrogram Datasets/Overlapping/size"+str(size)+"_overlap10/"+ label)

              # 300 rows in a new data-frame  
        full_flight= pd.read_csv(DATA_PATH+ filename+".csv")
        split_dfs = [full_flight.loc[i:i+size-1,:] for i in range(0, len(full_flight),size-10) if i <len(full_flight)-10]
   
        while len(split_dfs[-1])<size:
            df2 = pd.DataFrame([[0]*split_dfs[-1].shape[1]],columns=split_dfs[-1].columns)
            split_dfs[-1] = split_dfs[-1].append(df2, ignore_index=True)
            #del df2  
            
        if label=="Quadrotor x":
            quadx_count+=len(split_dfs)

        if (label!="Quadrotor x") | (quadx_count<20000):
            for flight_segment, segment_id in zip(split_dfs, range(0,len(split_dfs)-1)):

            
                flight= _movement_flight_features(flight_segment)
                D= _flight_spectrogram(flight)
                if plot==True:
                    _plot_spectrogram(D, filepath= "C:/Users/pacofarina/davidUAVClassification-2/Data/Spectrogram Datasets/Overlapping/size"+str(size)+"_overlap10/"+label+"/"+filename+"_"+str(segment_id)+label)
                
                #overlapping_df=overlapping_df.append({"spectrogram": D, "label": label, "ULG":ulg},ignore_index=True)
                ## Free up memory
                #del flight, D            
            #free memory
            #del filename,label,ulg
            #Update progress bar    
        pbar.update(1)
    overlapping_df.to_csv("../Data/Spectrogram Datasets/Overlapping/size"+str(size)+"_overlap10/spectrograms.csv")
    pbar.close()
    return filtered_df

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






def main():
    
    
    #overlapping(size=300, plot=True)
    _combine_sub_classes()





if __name__ == "__main__":
    main()










