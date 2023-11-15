################################
# This script contains utilities used for the spectrogram generation
# 
# @author: David Sanchez <davsanch@inf.uc3m.es>
# @author: Daniel Amigo <damigo@inf.uc3m.es>
# @author: Paco Fariña  <franciscofarinasalguero@gmail.com>
################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import librosa # Efficient spectrogram generation
import librosa.display
import  pywt   # Scaleogram generation
import warnings

import matplotlib.cbook as cbook


def _calculateMagnitude(xcomp,ycomp,zcomp):


    return np.sqrt(np.power(xcomp,2)+np.power(ycomp,2)+np.power(zcomp,2))

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
 
def _create_scaleogram(signal: np.ndarray) -> np.ndarray:
    """Creates scaleogram for signal, and returns it.
    The resulting scaleogram represents scale in the first dimension, time in
    the second dimension, and the color shows amplitude.
    """
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


def _movement_flight_features(flight):
      
    flight["vmag"]= _calculateMagnitude(flight["vx"],flight["vy"],flight["vz"])

    flight=_calculateAcceleration(flight)

    flight["amag"]=_calculateMagnitude(flight["ax"],flight["ay"],flight["az"])
    flight["amag"].iloc[0]=flight["amag"].iloc[1]
    flight.isnull().sum(axis = 0)

    return flight

def _flight_spectrogram(flight,sr=0.1, hop_length = 1,n_fft = 128):

    return librosa.amplitude_to_db( np.abs(librosa.stft(flight["amag"].to_numpy(), hop_length=hop_length, n_fft=n_fft)), ref=np.max)


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

    


def _filter_by_duration( duration: int, df: pd.DataFrame ):

    filtered_df= df[[df["Duration"]>= duration]]

    return df
