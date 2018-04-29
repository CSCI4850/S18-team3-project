# Team Pandemonium

## Installing dependences:  
### Install the python package manager PIP  

```python<3> get-pip.py``` or ```sudo easy_install pip```  

__Most system use Python 2.7 as the default python interpreter.__ Be sure to check which python you're using via ```python --version``` if you are unsure of your system's default python interpreter.  You more than likely need to use ```python3``` instead of just ```python```.  PIP might be available via brew for macOS: ```brew install pip3``` is likely to work.  There maybe a way to get ```PIP``` via packages such as ```sudo apt install python3-pip``` in most Ubuntu/apt based systems.  You may also use ```pip --version``` to display the location, version of pip, and the associated python version to verify your setup.

### Install virutalenv
install virtualenv using pip:  

```pip<3> install virtualenv```

virtualenv is used to create and manage environments for different python projects.  Use virtualenv to create a virtual environment by issuing:

```virtualenv env```

to create a folder named env which will store relevant python related files.  

__NOTE:__ If you are in a virtual environment -- do not use ```sudo```, as this will not install into the virtual environment, but the system's environment instead.

### Switching environments
Use ```source env/bin/activate``` to load your environment.  ```env``` is a placeholder for your environment name.  

use ```deactivate``` to exit out of the virtual environment.  

To verify which environment you're in, use ```which pip```.  if you see that the pip location is in your environment folder (env), then you are in your virtual environment.  Also notice that in virtualenv, python3 is now the default interpreter which makes life much easier.  Check using ```python --version``` and notice the __lack__ of 3 at the end.

### Install packages
Installing python packages can be done by using the `requirements.txt`:  
```pip install -r requirements.txt```

## Data preprocessing and preparation

### NLTK parts of speech data  

The NLTK library requires you download corpora data for our part of speech tagging.  You can download and install these data via a python interpreter:

```
>>> import nltk  
>>> nltk.download()
```

A dialog window will pop-up after the function call which allows you to select and install data.  The perceptron tagger might be the only nltk data required, but in our development we used all nltk data.

### Downloading cartoon script data  
Our project uses the American cartoon show, Rick and Morty, to create our training corpus. The script can be aquired by running 'utils/preprocess.py' FROM THE ROOT DIRECTORY, which will download and format the data, placing it in '/data/train/cleaned/simple.txt'
