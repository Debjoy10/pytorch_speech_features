pytorch_speech_features
=======================

A simple PyTorch reimplementation of library
`python_speech_features <https://github.com/jameslyons/python_speech_features>`__.

Uses
----

-  **Great for Intepretability experiments** - All audio processing
   operations can be performed and the results can be backpropagated to
   the original signal tensor.
-  **Supports Hybrid Model Design** - Parametric operations at different
   stages of audio processing.

`Example
use <https://github.com/Debjoy10/pytorch_speech_features/blob/main/demo.ipynb>`__

Installation
------------

   Install from GitHub

::

   git clone https://github.com/Debjoy10/pytorch_speech_features
   python setup.py develop

Usage
-----

Functions same as python_speech_features (`Refer to its documentation here <https://python-speech-features.readthedocs.io/en/latest/>`__).


   Instead of input signal as list / numpy array, pass torch tensor (both ‘cpu’ and ‘cuda’ supported!!).

::

See example use given above.

Testing
-------
Two things to test for pytorch_speech_features operations -   


-  Similarity to python_speech_features outputs. 
-  Gradient correctness via Autograd Gradcheck.

Find the testing python notebook here -
'''''''''''''''''''''''''''''''''''''''

|Open In Colab|

Citation
--------

*TODO*

References
----------

-  Python_speech_features library -
   `Link <https://github.com/jameslyons/python_speech_features>`__
-  Sample english.wav -
   `Link <http://voyager.jpl.nasa.gov/spacecraft/audio/english.au>`__

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1Tyizr62YGi5-CR-o-tawV6pu68JT3DOF?usp=sharing
