# pytorch_speech_features

[![DOI](https://zenodo.org/badge/470451115.svg)](https://zenodo.org/badge/latestdoi/470451115)

A simple PyTorch reimplementation of library [python_speech_features](https://github.com/jameslyons/python_speech_features). 

## Uses

* **Great for Intepretability experiments** - All audio processing operations can be performed and the results can be backpropagated to the original signal tensor.
* **Supports Hybrid Model Design** - Parametric operations at different stages of audio processing.  

[Example use](https://github.com/Debjoy10/pytorch_speech_features/blob/main/demo.ipynb)

## Installation
> Install from PyPI
```
pip install pytorch-speech-features
```

> Install from GitHub
```
git clone https://github.com/Debjoy10/pytorch_speech_features
python setup.py develop
```  

## Usage
Functions same as python_speech_features ([Refer to its documentation here](https://python-speech-features.readthedocs.io/en/latest/)).  
> Instead of input signal as list / numpy array, pass tensor (both 'cpu' and 'cuda' supported!!). 

See example use given above.  

Supported features:
* Mel Frequency Cepstral Coefficients
* Filterbank Energies
* Log Filterbank Energies
* Spectral Subband Centroids

## Testing
Two things to test for pytorch_speech_features operations - 
1. Similarity to python_speech_features outputs.
2. Gradient correctness via Autograd Gradcheck. 

##### Find the testing python notebook here - 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tyizr62YGi5-CR-o-tawV6pu68JT3DOF?usp=sharing)

## Citation
```
@misc{https://doi.org/10.5281/zenodo.8021586,
  doi = {10.5281/ZENODO.8021586},
  url = {https://zenodo.org/record/8021586},
  author = {{Debjoy Saha}},
  title = {Debjoy10/pytorch_speech_features: Release v0.0.1},
  publisher = {Zenodo},
  year = {2023},
  copyright = {Open Access}
}
```

## References
* Python_speech_features library - [Link](https://github.com/jameslyons/python_speech_features)
* Sample english.wav - [Link](http://voyager.jpl.nasa.gov/spacecraft/audio/english.au)
