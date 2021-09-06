'''Isolate formants from a voice sample.

Given a 1 second voice sample of a vowel formant, this script will return
the 10 loudest resonant frequencies found in that sample. These frequencies
can also be used to synthesize a voice from the provided voice sample.

This script is used to explore acoustic phonetics, specifically, in how a 
voice can be expressed as a collection of sinusoids. Although an infinite 
set of sinusoids is needed to completely replicate a voice, This script 
demonstrates how formants are a critical component for speaking vowels and 
how differences in the amount of formants collected can change the quality 
of voice a sample. 

Required modules: scipy, numpy, matplotlib

Input and output audio files must be .wav files

This script can be imported as a module and contains the following functions:

    _getVoiceSampleProperties - Retrieves data about voice sample.
    _getSpectrumDataFromAudacity - Gets freq and amplitudes from Audacity 
                                   Spectrum export.
    getLocalMaxima - Returns n largest local Maximas in a normalized spectrum.
    getNormalizedSpectrum - Returns the formant frequencies and amplitudes 
                            for a voice sample.
    generateSynthVoice - Returns an array defining the resultant signal of 
                         a sum of sine waves.
'''

from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, fft, fftfreq
from scipy.io import wavfile
import numpy as np
from scipy.signal import argrelextrema
import argparse
import pathlib

def _getVoiceSampleProperties(voiceSample):
    ''' Retrieves data about voice sample for debugging purposes. 
    
    Parameters
    ----------
        voiceSample: str
            File name of wav file that requires analysis
    '''

    sampleRate, data = wavfile.read(voiceSample)
    # Only process the first channel in the sample: 
    if data.ndim > 1:
        data = data[:,0]
    # Convert 16 Bit signal to values between -1 and 1.
    data = np.divide(data, data.max())
    # Get signal spectrum and normalize the intensity between 0 and 1
    # 1 Represents the loudest frequency
    yf = rfft(data)
    yf = np.abs(yf)
    yf = np.divide(yf, yf.max())
    # Get frequency domain from fourier transform
    xf = rfftfreq(data.size, 1 / sampleRate)
    # Filter out all frequencies below threshold
    threshold = 0.01
    yf[yf<threshold] = 0
    print('Sample Rate: {0}'.format(sampleRate))
    print('data size: {0}'.format(data.size))
    print('data shape: {0}'.format(data.shape   ))
    print('Input array is imaginary?: {0}'.format(np.any(np.iscomplex(data))))
    print('yf size: {0}'.format(yf.size))
    print('yf shape: {0}'.format(yf.shape))
    print('xf size: {0}'.format(xf.size))
    print('Output array is imaginary?: {0}'.format(np.any(np.iscomplex(yf))))
    print('Sample Points: {0}'.format(yf[2000:2010]))
    print('plot')
    plt.plot(xf[:4000], yf[:4000])
    plt.show()

def _getSpectrumDataFromAudacity(spectrumFile):
    ''' Gets freq and amplitudes from Audacity Spectrogram export (For debug)
    
    Parameters
    ----------
        spectrumFile: str
            Name of txt file containing Audacity Spectrogram export
    '''

    inputSpectrumDict = {}
    with open(spectrumFile, 'r') as f:
        for curLine in f.readlines()[1:]:
            (curKey, curVal) = curLine.split('\t')
            curFreq = float(curKey)
            curDBLevel = float(curVal.replace('\n', ''))
            if curFreq < 4000:
                inputSpectrumDict[curFreq] = curDBLevel
    print(inputSpectrumDict)

def getLocalMaxima(freqArray, ampArray, threshold, maximaCount):
    ''' Returns n largest local Maximas in a normalized spectrum 
    
    Parameters
    ----------
        freqArray: numpy.ndarray
            1D array containing frequency domain of spectrum
        ampArray: numpy.ndarray
            1D Array of normalized amplitudes of spectrum
        threshold: int
            Minimum amplitude required to be considered for local 
            maxima identification
        maximaCount: int 
            Number of n largest local maximas found in spectrum
    
    Returns
    -------
        outFreq: numpy.ndarray
            1D array of frequences with n largest local maxima in sample
        outAmp: numpy.ndarray
            1D array of amplitudes corresponding to the frequencies in outFreq
    '''

    # Filter out all frequencies below threshold
    ampArray[ampArray<threshold] = 0
    # Find frequencies that are local maximas
    maximaInd = argrelextrema(ampArray, np.greater)[0]
    maximaArray = ampArray[maximaInd]
    # Get n loudest frequencies to define characteristic frequencies
    sortedNMaxInd = np.argsort(maximaArray)[-maximaCount:]
    charFreq = maximaInd[sortedNMaxInd]
    # Sort frequencies and amplitudes by increasing frequency
    # To match order of appearance in spectrum graph
    outFreq = freqArray[charFreq]
    ascFreqInd = np.argsort(outFreq)
    outFreq = outFreq[ascFreqInd]
    outAmp = ampArray[charFreq]
    outAmp = outAmp[ascFreqInd]
    return (outFreq, outAmp)

def getNormalizedSpectrum(voiceSample, binCount, maxFreq=None):
    ''' Returns the formant frequencies and amplitudes for a voice sample

    Parameters
    ---------- 
        voiceSample: str
            File name or path .wav file containing 1 second voice sample
        binCount: int
            Number of bins to catagorize frequencies.
        maxFreq: int
            Largest frequency to be analyzed in voice sample
    
    Returns
    -------
        xfBinned: numpy.ndarray
            1D array of wieghted frequencies found in voice sample 
        yfBinned: numpy.ndarray
            1D array of amplitudes corresponding to weighted frequencies

    Notes
    -----
    Example: 500 bins with a max freqeuncy of 4000 Hz will return 
    an array of 500 frequency bins with amplitudes ranging between 
    0 and 1 (1 representing the largest amplitude recorded in the 
    sample). 8 Frequencies are assigned to each bin (e.g. 0-7, 8-15
    ... 3991-3999). The weighted average is found for the frequencies
    and amplitudes for each bin such that each bin has 1 effective 
    frequency and effective amplitude associated with it.  
    '''

    sampleRate, data = wavfile.read(voiceSample)
    # Assume mono sound so only process the first channel in the sample: 
    if data.ndim > 1:
        data = data[:,0]
    # Convert 16 Bit signal to values between -1 and 1.
    data = np.divide(data, data.max())
    # Get signal spectrum and normalize the intensity between 0 and 1
    # 1 Representing the loudest frequency
    yf = rfft(data)
    yf = np.abs(yf)
    yf = np.divide(yf, yf.max())
    # Get frequency domain from fourier transform
    xf = rfftfreq(data.size, 1 / sampleRate)
    # Truncate the spectrum up to the maximum Frequency 
    # (4000 Hz for most speech)
    xf = xf[:maxFreq]
    yf = yf[:maxFreq]
    # Classify frequencies into bins to smooth out spectrum graph
    bins = np.arange(binCount) * (yf.size / binCount) # [0, 40, 80 ... 3960]
    xfIndForBins = np.digitize(xf, bins) # [1,2,3, ... etc]
    # Get the weighted frequency and amplitude associated with each bin
    # to smooth out spectrum graph for finding local maximas later on
    xfBinnedList = []
    yfBinnedList = []
    ## Add first element to BinnedLists outside for-loop 
    ## to handle issue with "0" Hz frequency
    maskArray = np.in1d(xfIndForBins, 1) 
    binYfs = yf * maskArray
    binYfs = binYfs[np.where((binYfs)!=0)]
    binXfs = xf * maskArray
    binXfs = binXfs[np.where((binXfs)!=0)]
    binXfs = np.insert(binXfs, 0, 0)
    xfWeighted = np.average(binXfs, weights=binYfs)
    xfBinnedList.append(xfWeighted)
    yfWeighted = np.average(binYfs)
    yfBinnedList.append(yfWeighted)
    ## Loop through remaining bins 
    ## to get the rest of the weighted frequencies and amplitudes
    for curInd in range(2, bins.size):
        maskArray = np.in1d(xfIndForBins, curInd)
        binYfs = yf * maskArray
        binYfs = binYfs[np.where((binYfs)!=0)]
        binXfs = xf * maskArray
        binXfs = binXfs[np.where((binXfs)!=0)]
        xfWeighted = np.average(binXfs, weights=binYfs)
        xfBinnedList.append(xfWeighted)
        yfWeighted = np.average(binYfs)
        yfBinnedList.append(yfWeighted)
    xfBinned = np.array(xfBinnedList)
    yfBinned = np.array(yfBinnedList)
    # Normalize the intensity of yfBinned between 0 and 1
    # 1 Represents the loudest frequency
    yfBinned = np.divide(yfBinned, yfBinned.max())
    return (xfBinned, yfBinned)

def generateSynthVoice(freqMaximaArray, ampMaximaArray, duration, sampleRate):
    ''' Returns an array defining the resultant signal of a sum of sine waves
    
    Parameters
    ----------
    freqMaximaArray: numpy.ndarray 
        1D array of frequencies of sine waves that will be summed
    ampMaximaArray: numpy.ndarray 
        Relative amplitudes of the above sine waves (1 being the loudest)
    duration: int
        The length of the generated signal in seconds 
    sampleRate: int 
        The number of points defined for the resultant signal in one second
    
    Returns
    -------
    sampleRate: int
        The number of points defined for the resultant signal in one second
    ampRange: numpy.ndarray
        The 

    Notes
    -----
    Since a human voice can be defined as the summation of an infinite set of 
    sine waves at various frequencies and amplitudes, this function attempts to
    synthesize an artificial voice from a finite collection of sine waves by 
    using the results of the getLocalMaxima function. The output of this 
    function can be used to generate a .wav file.
    '''
    
    timeDomain = np.linspace(0, duration, sampleRate * duration, endpoint=False)
    # Round maxima frequencies to nearest integer (Because I like round numbers)
    freqMaximaArray = np.round(freqMaximaArray)
    # Initialize output array by using first elements in frequency and amplitude arrays
    ampRange = ampMaximaArray[0] \
                * np.sin((2 * np.pi) \
                * freqMaximaArray[0] * timeDomain)
    # Add signals to output array for remaining frequencies and amplitudes
    for curInd in range(1, freqMaximaArray.size):
        ampRange += ampMaximaArray[curInd] \
                    * np.sin((2 * np.pi) \
                    * freqMaximaArray[curInd] * timeDomain)
    # Normalize the output signal between -1 and 1 to confirm to .wav standard 
    ampRange = (ampRange / np.max(np.abs(ampRange)))
    # Reduce output signal intensity to 80% of max to reduce speaker strain
    # and simplify conversion to 16 bit array
    ampRange = ampRange * 0.8
    # Convert to 16 Bit signal
    ampRange = (ampRange * 32768).astype(np.int16)
    return (sampleRate, ampRange)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fileName', metavar='f', 
        help='Path to .wav file containing 1 second voice sample'
    )
    parser.add_argument('-o', dest='outFile', help='Path to output .wav file')
    args = parser.parse_args()
    voiceSampleFile = pathlib.Path(args.fileName)
    if voiceSampleFile.is_file():
        (freqArray, ampArray) = getNormalizedSpectrum(args.fileName, 500, 4000)
        (freqMaximaArray, ampMaximaArray) = getLocalMaxima(freqArray, ampArray, 0.01, 10)
        print('Formants of raw voice sample: {0}'.format(np.round(freqMaximaArray)))
        print('Amplitudes of formants: {0}'.format(np.round(ampMaximaArray, 3)))
        if args.outFile is not None:
            synthNewSampleRate, synthNewData = generateSynthVoice(freqMaximaArray, ampMaximaArray, 1, 44100)
            wavfile.write(args.outFile, synthNewSampleRate, synthNewData)
            print('Synthesized voice sample created.')
        plt.plot(freqArray, ampArray, label='Voice Spectrum')
        plt.plot(freqMaximaArray, ampMaximaArray, marker='o', label='Local Maxima')
        plt.legend()
        plt.show()