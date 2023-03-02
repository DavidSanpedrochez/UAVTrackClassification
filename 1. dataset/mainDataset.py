#IMPORTS
import time
import os
from os import listdir
from os.path import isfile, join
import pathlib
import shutil
import multiprocessing
import psutil
from parallelUlogs2CSV import parallelUlogs2CSV
import traceback
import pandas as pd
import traceback

############
# Manda un aviso de una sneaker por Telegram
############
## para el bot de telegram
import telebot
def messageTelegramBot(text):
    TOKEN   = '' # Ponemos nuestro Token generado con el @BotFather
    CHAT_ID = ''
    tb      = telebot.TeleBot(TOKEN) # Combinamos la declaración del Token con la función de la API
    tb.config['api_key'] = TOKEN
    # manda la foto, mensaje con modelo, tallas, descuento y precio, y link
    if tb.get_me() is not None: # si funciona la API
        tb.send_message(CHAT_ID, text)

if __name__ == "__main__":

    arrDesiredFiles = [
        "_vehicle_global_position",
        "_vehicle_rates_setpoint",
        "_vehicle_attitude_setpoint",
        "_vehicle_local_position_setpoint",
        "_vehicle_local_position_groundtruth",
        "_vehicle_global_position_groundtruth",
        "_vehicle_attitude_grountruth",
        "_vehicle_angular_acceleration",
        "_estimator_local_position",
        "_estimator_global_position",
        "_vehicle_local_position",
        "_vehicle_land_detected",
        "_vehicle_gps_position",
        "_vehicle_angular_velocity",
        "_trajectory_setpoint",
        "_ekf2_timestamps",
        "_sensor_gps",
        "_vehicle_attitude",
        "_vehicle_acceleration",
        "_takeoff_status",
        "_sensor_combined",
        "_home_position",
        "_sensor_baro",
        "_estimator_attitude",
    ]

    parallelize = True # True to execute in parallel, False to execute sequentially
    listOriginal = ["batch0", "batch1", "batch2", "batch3", "batch4", "batch4.5", "batch5", "batch6"]
    summaryPath  = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "DatasetGeneration")
    combineBatches = False   # After processing all batches separatedly, combine them into a single summarized CSV file (useful for parallel processing)
    mergeFiles     = False   # Merge all the CSV files folders into a single folder (useful for parallel processing)
    processing     = False   # Perform the dataset generation process (True) or skip it (False) and use the already generated files
    crossTable     = False   # Enhance the summary CSV file by crossing it with the airframe table. It will generate additional columns
    finalDataset   = True    # 

    if processing == True:
        ####### START PROCESSING #######
        for k in range(len(listOriginal)):

            start=time.time()

            #PATHS
            originalPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "DatasetGeneration", listOriginal[k])
            modifiedPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "DatasetGeneration", "proc_"+listOriginal[k])
            if os.path.exists(modifiedPath): # delete the folder if it exists
                shutil.rmtree(modifiedPath)
            os.makedirs(modifiedPath)

            # get files in the folder
            fileList = [f for f in listdir(originalPath) if isfile(join(originalPath, f))]

            if parallelize:
                # paralellize execution
                numThreads = psutil.cpu_count()
                numThreads = 16 # Number of threads to use

                numFiles = len(fileList) # Number of files to process
                numFilesPerThread = int(numFiles/numThreads) # Number of files to process per thread
                print("Number of files to process: ", numFiles)
                print("Number of files per thread: ", numFilesPerThread)
                threads = []
                for i in range(numThreads):
                    # get files to process in this thread
                    if i == numThreads-1: filesToProcess = fileList[i*numFilesPerThread:]        # last thread processes the remaining files
                    else: filesToProcess = fileList[i*numFilesPerThread:(i+1)*numFilesPerThread] # other threads process the files assigned to them
                    # Create the thread
                    t = multiprocessing.Process(target=parallelUlogs2CSV, args=(originalPath, modifiedPath, i, filesToProcess, arrDesiredFiles))
                    threads.append(t)
                    t.start()

                # Wait for all threads to finish
                for t in threads:
                    t.join()

                # Merge the results
                for i in range(numThreads):
                    filename = os.path.join(modifiedPath, f"DroneModel_{i}.csv")
                    # read the CSV with Pandas
                    df = pd.read_csv(filename, header=0)

            else: # sequential execution, for debugging purposes
                print("Sequential execution")
                #fileList = fileList[0:50]
                parallelUlogs2CSV(originalPath, modifiedPath, numThread=0, filesToProcess=fileList, arrDesiredFiles=arrDesiredFiles)

            print("DONE")
            messageTelegramBot(f"TERMINACIÓN")

            if parallelize:
                # Al finalizar, leer los resultados de cada uno y generar la tabla definitiva explicando las características de cada trayectoria
                # Merge the results
                combinedDF = pd.DataFrame()
                for i in range(numThreads):
                    filename = os.path.join(modifiedPath, f"DroneModel_{i}.csv")
                    # read the CSV with Pandas
                    df = pd.read_csv(filename, header=0)
                    combinedDF = pd.concat([combinedDF, df], axis=0)
                    if i == 0:
                        # set header to CombinedDF
                        combinedDF.columns = df.columns
                    # Remove the individual files
                    os.remove(filename)
                
                # Save the combined dataframe
                combinedDF.to_csv(os.path.join(summaryPath, listOriginal[k]+".csv"), index=False)
                # Calculate the mean of the combined dataframe
                meanDF = combinedDF.mean(axis=0)
                print(meanDF)
            
            # Return the processing time
            end = time.time()
            processingTime = end-start
            print(f"Processing time: {processingTime} seconds")

    ######################### COMBINE BATCHES IN ONE CSV FILE#########################
    if combineBatches == True:
        # Merge batches
        combinedDF = pd.DataFrame()
        for k in range(len(listOriginal)):
            filename = os.path.join(summaryPath, listOriginal[k]+".csv")
            # read the CSV with Pandas
            df = pd.read_csv(filename, header=0)
            combinedDF = pd.concat([combinedDF, df], axis=0)
            if k==0:
                # set header to CombinedDF
                combinedDF.columns = df.columns
        # Save the combined dataframe
        combinedDF.to_csv(os.path.join(summaryPath, "combination"+".csv"), index=False)

    ################### MERGE TRAJECTORY CSVS IN ONE FOLDER ###################
    if mergeFiles == True:
        # Create the merged folder
        mergedPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "merged")
        if os.path.exists(mergedPath): # delete the folder if it exists
            shutil.rmtree(mergedPath)
        os.makedirs(mergedPath)

        # Merge trajectory CSVs
        for k in range(len(listOriginal)):
            modifiedPath = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "proc_"+listOriginal[k])
            # get files in the folder
            fileList = [f for f in listdir(modifiedPath) if isfile(join(modifiedPath, f))]
            for i in range(len(fileList)): # move files
                if fileList[i].endswith(".csv"):
                    shutil.move(os.path.join(modifiedPath, fileList[i]), os.path.join(mergedPath, fileList[i]))

    ######################### CROSS WITH AIRFRAME TABLE #########################
    if crossTable == True:
        from addColumnsAirframes import crossSummaryTableWithAirframe
        crossSummaryTableWithAirframe(filesPath=summaryPath, summary="combination", airframeTable="Airframes PX4", outputName="combinationWithAirframe")

    if finalDataset == True:
        from addColumnsAirframes import finalDataset
        listOriginal = ["batch0", "batch1", "batch2", "batch3", "batch4", "batch4.5", "batch5", "batch6"]
        summaryPath  = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "DatasetGeneration")
        inputPath    = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "DatasetGeneration", "finalisimov2.csv")
        outputFolder = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "DatasetGeneration", "finalisimov2")
        mergedPath   = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "Data", "DatasetGeneration", "merged")

        finalDataset(inputCSVTable=inputPath, trajectoriesFolder=mergedPath, outputFolder=outputFolder)
       
    # END END