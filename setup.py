try:
    from setuptools import setup #enables develop
except ImportError:
    from distutils.core import setup

setup(name='pytorch_speech_features',
      version='0.0.1',
      description='PyTorch Speech Feature extraction',
      author='Debjoy Saha',
      author_email='sahadebjoy10@gmail.com',
      license='MIT',
      url='https://github.com/Debjoy10/pytorch_speech_features',
      packages=['pytorch_speech_features'],
      install_requires=[
        'numpy',
        'scipy',
        'torch',
      ]
    )