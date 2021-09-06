# formantextractor

## Synopsis

Isolate formants from a voice sample.

## Description

Given a 1 second voice sample of a vowel formant, this script will return the 10 loudest resonant frequencies found in that sample. These frequencies can also be used to synthesize a voice from the provided voice sample.

This script is used to explore acoustic phonetics, specifically, in how a voice can be expressed as a collection of sinusoids. Although an infinite set of sinusoids is needed to completely replicate a voice, This script demonstrates how formants are a critical component for speaking vowels and how differences in the amount of formants collected can change the quality of voice a sample. 

### Prerequisites

* Python 3.9
* Module Dependencies: scipy, numpy, matplotlib

### How to Use

To return the 10 loudest formants in a voice sample with corresponding spectrum plot:

```
py formantextractor.py inputVoiceSample.wav
```

To return the 10 loudest formants in a voice sample  with spectrum plot and generate a synthesized voice from these formants:

```
py formantextractor.py -o outputSynthSample.wav inputVoiceSample.wav
```

Refer to the Examples folder in this repo for additional examples.

### Additional Notes

It is recommended to use the audio software Audacity to record the 1 second raw voice samples. One of Audacity's features is to plot the spectrum of a provided voice sample. This feature was used to verify the results of the python script. 