try:
    from setuptools import setup #enables develop
except ImportError:
    from distutils.core import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='pytorch_speech_features',
      version='0.0.3',
      description='PyTorch Speech Feature extraction',
      long_description=long_description,
      long_description_content_type='text/markdown',
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