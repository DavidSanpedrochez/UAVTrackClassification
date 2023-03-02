
import pandas as pd
import numpy as np
import math
import pathlib
import shutil

#######
# Read two Pandas DataFrames from CSV files
def crossSummaryTableWithAirframe(filesPath="", summary="", airframeTable="", outputName=""):

    airFramesPath = pathlib.Path(filesPath, airframeTable+'.csv')
    airframes = pd.read_csv(airFramesPath)
    mergePath = pathlib.Path(filesPath, summary+'.csv')
    merge     = pd.read_csv(mergePath)

    # add columns to merge
    merge['ModelName']  = ''
    merge['Simulation'] = ''
    merge['General']    = ''
    merge['Class']      = ''
    merge['Num arms']   = ''

    #########
    for i, row in merge.iterrows():
        a = airframes.loc[airframes['ID'] == row['Model']]
        dictNew = {}
        if len(a) > 0:
            # Copy the row from the first DataFrame
            merge.at[i,'ModelName'] = a['PX4 Code Name'].values[0].replace('\t', '')
            merge.at[i,'General']   = a['General'].values[0]
            merge.at[i,'Class']     = a['Class'].values[0]
            merge.at[i,'Num arms']  = a['Num arms'].values[0]
            if a['Simulation'].values[0] == 'X':
                merge.at[i,'Simulation'] = 'X'
            else:
                merge.at[i,'Simulation'] = '-'
        else:
            merge.at[i,'ModelName']  = '??'
            merge.at[i,'General']    = '??'
            merge.at[i,'Class']      = '??'
            merge.at[i,'Num arms']   = '??'
            merge.at[i,'Simulation'] = '??'
        
    # Write the new DataFrame to a CSV file
    outputPath = pathlib.Path(filesPath, outputName+'.csv')
    merge.to_csv(outputPath, index=False)

#####
# This function receives a final and handcrafted table with the ulog IDs to build the dataset
# It also receives the path to the folder with the ulog files
# It creates a new folder with the CSV files of the final dataset
def finalDataset(inputCSVTable="", trajectoriesFolder="", outputFolder=""):
    # Read the CSV file with the final table
    finalTable = pd.read_csv(inputCSVTable)

    # Create the folder to store the CSV files
    if pathlib.Path(outputFolder).exists(): # IF EXISTS, DELETE IT
        shutil.rmtree(outputFolder)
    pathlib.Path(outputFolder).mkdir(parents=True, exist_ok=True)

    # Iterate over the CSV table
    for i, row in finalTable.iterrows():
        # Show percentage
        if i % 100 == 0:
            print(f"{i/len(finalTable)*100}% of the files copied")

        # Get the ID of the trajectory
        trajectoryID = row['ULG']
        # Find this ULG file in the folder with the ulog files
        found = False
        newThisFile = pathlib.Path(trajectoriesFolder, trajectoryID+".csv")
        if newThisFile.exists():
            # Copy the file to the new folder
            newPath = pathlib.Path(outputFolder, trajectoryID+".csv")
            shutil.copy(newThisFile, newPath)
            found = True
        if not found:
            print("File not found: ", trajectoryID+".ulg")
        