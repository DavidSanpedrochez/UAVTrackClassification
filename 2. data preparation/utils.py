################################
# This script contains utilities used for the spectrogram generation
# 
# @author: David Sanchez <davsanch@inf.uc3m.es>
# @author: Daniel Amigo <damigo@inf.uc3m.es>
# @author: Paco Fariña  <franciscofarinasalguero@gmail.com>
################################
#############################################################
#                       IMPORTS                             #
#############################################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import librosa # Efficient spectrogram generation
import librosa.display
import  pywt   # Scaleogram generation
import warnings

import matplotlib.cbook as cbook

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
# Calculate vector magnitude from x,y,z components
# *******  [INPUT] xcomp: x component of the vector
# *******  [INPUT] ycomp: y component of the vector
# *******  [INPUT] zcomp: z component of the vector
# ******* [OUTPUT] magnitude of the vector
# *****************************************************************************************
def _calculateMagnitude(xcomp,ycomp,zcomp):
    return np.sqrt(np.power(xcomp,2)+np.power(ycomp,2)+np.power(zcomp,2))

# *****************************************************************************************
# Calculate acceleration from velocity
# *******  [INPUT] df: dataframe with the velocity components
# ******* [OUTPUT] df: dataframe with the acceleration components added
# *****************************************************************************************
def _calculateAcceleration( df: pd.DataFrame) -> pd.DataFrame:
    
    #we want it in m/s^2 and timestamp is in microseconds
    df["ax"]= df["vx"].diff()/(df["timestamp"].diff()/np.power(10,6))
    df["ay"]= df["vy"].diff()/(df["timestamp"].diff()/np.power(10,6))
    df["az"]= df["vz"].diff()/(df["timestamp"].diff()/np.power(10,6))

    #replace nan in first position by next value
    df["ax"].iloc[0]=df["ax"].iloc[1]
    df["ay"].iloc[0]=df["ax"].iloc[1]
    df["az"].iloc[0]=df["ax"].iloc[1]
    
    return df
 
# *****************************************************************************************
# Creates scaleogram for signal, and returns it.
# *******  [INPUT] signal: signal to create scaleogram from
# ******* [OUTPUT]  The resulting scaleogram represents scale in the first dimension, time in
#                       the second dimension, and the color shows amplitude.
# *****************************************************************************************
def _create_scaleogram(signal: np.ndarray) -> np.ndarray:

    n = len(signal)  # 128

    # In the PyWavelets implementation, scale 1 corresponds to a wavelet with
    # domain [-8, 8], which means that it covers 17 samples (upper - lower + 1).
    # Scale s corresponds to a wavelet with s*17 samples.
    # The scales in scale_list range from 1 to 16.75. The widest wavelet is
    # 17*16.75 = 284.75 wide, which is just over double the size of the signal.
    scale_list = np.arange(start=0, stop=n) / 8 + 1  # 128
    wavelet = "gaus1"
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0]

    return scaleogram


# *****************************************************************************************
# Generates features from a given track
# *******  [INPUT] track: dataframe with the track
# ******* [OUTPUT] track: dataframe with the track and the features calculated:
#                       - vmag: velocity magnitude
#                       - ax: acceleration in x component
#                       - ay: acceleration in y component  
#                       - az: acceleration in z component 
#                       - amag: acceleration magnitude     
# *****************************************************************************************
def _movement_flight_features(track):
      
    track["vmag"]= _calculateMagnitude(track["vx"],track["vy"],track["vz"])

    track=_calculateAcceleration(track)

    track["amag"]=_calculateMagnitude(track["ax"],track["ay"],track["az"])
    track["amag"].iloc[0]=track["amag"].iloc[1]
    track.isnull().sum(axis = 0)

    return track

# *****************************************************************************************
# Generates spectrogram from a given flight
# *******  [INPUT] flight: dataframe with the flight
# *******  [INPUT] sr: sampling rate
# *******  [INPUT] hop_length: number of samples between successive frames
# *******  [INPUT] n_fft: length of the FFT window
# ******* [OUTPUT] spectrogram of the flight
# *****************************************************************************************
def _flight_spectrogram(flight,sr=0.1, hop_length = 1,n_fft = 128):
    return librosa.amplitude_to_db( np.abs(librosa.stft(flight["amag"].to_numpy(), hop_length=hop_length, n_fft=n_fft)), ref=np.max)


# *****************************************************************************************
# Plot a given spectrogram 
# *******  [INPUT] D: spectrogram to plot
# *******  [INPUT] filepath: path to save the plot
# *******  [INPUT] sr: sampling rate
# *******  [INPUT] fmin: minimum frequency
# *******  [INPUT] fmax: maximum frequency
# *******  [INPUT] y_axis: frequency axis scale
# *******  [INPUT] hop_length: number of samples between successive frames
# *******  [INPUT] n_fft: length of the FFT window
# *******  [INPUT] auto_aspect: if True, set the aspect ratio automatically
# *******  [INPUT] bins_per_octave: number of bins per octave
# *******  [INPUT] cmap: colormap
# ******* [OUTPUT] saved figure at filtepath
# *****************************************************************************************
def _plot_spectrogram(D,filepath,sr=0.1,fmin = None,  fmax = None, y_axis = "linear", hop_length = 1,n_fft = 43,auto_aspect = False,bins_per_octave = 12,cmap = 'jet'): 

    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, y_axis=y_axis, sr=sr,
                                    hop_length=hop_length, x_axis='time', ax=ax, cmap=cmap, bins_per_octave=bins_per_octave,
                                    auto_aspect=auto_aspect)

    if fmin is not None:
            fmin0 = fmin
    else:
            fmin0 = 0
    if fmax is not None:
            fmax0 = fmax
    else:
            fmax0 = sr/2
    ax.set_ylim([fmin, fmax])
    #ax.legend().remove()
    plt.axis('off')
    dpi=100
    fig.set_size_inches(331/dpi, 333/dpi)
    plt.savefig(filepath+'.jpeg', bbox_inches='tight', dpi=dpi,pad_inches=0)
    plt.close()

    
# *****************************************************************************************
# Filter the dataframe by duration
# *******  [INPUT] duration: minimum duration of the flight
# *******  [INPUT] df: dataframe with the flights
# ******* [OUTPUT] filtered_df: dataframe with the flights with duration >= duration
# *****************************************************************************************
def _filter_by_duration( duration: int, df: pd.DataFrame ):

    filtered_df= df[[df["Duration"]>= duration]]

    return df
