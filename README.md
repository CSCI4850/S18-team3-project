# Team Pandemonium

# Installing dependences  
### Install the python package manager PIP  

```python<3> get-pip.py``` or ```sudo easy_install pip```  

__Most system use Python 2.7 as the default python interpreter.__ Be sure to check which python you're using with ```python --version``` if you are unsure of your system's default python interpreter.  You more than likely need to use ```python3```.  Pip might be available via brew for macOS: ```brew install pip3``` will likely work.  I also think there maybe a way to get pip via packages such as ```sudo apt install python3-pip``` in most Ubuntu/apt based systems.  You may also use ```pip --version``` to display which python pip will use.

### Install virutalenv

install virtualenv using pip:  

```pip<3> install virtualenv```

virtualenv is used to create and manage environments for different python projects.  Use virtualenv to create a virtual environment by using:

```virtualenv env```

__NOTE: If you are in a virtual environment -- do not, EVER, use ```sudo```, as this will not install into the virutal environment. In a virtual environment, the packages are installed in (this case) to the folder ```env```__

### Switching environments
Use ```source env/bin/activate``` to load your environment.  ```env``` is a placeholder for your environment name.  

use ```deactivate``` to exit out of your environment.  

To verify which environment you're in, use ```which pip```.  if you see that the pip location is in your environment folder (env), then you are in your virtual environment.  Also notice that in virtualenv, python3 is now the default interpreter which makes life much easier.  Check using ```python --version``` and notice the __lack__ of 3 at the end.

### Install packages

Then its as easy as:  
```pip install -r requirements.txt```



## TODO:
- Figure out project idea (text gen w/ tokens? find Waldo?)
- Distribute jobs among the five members
- Brush up on RNNs or CNNs
- Collect data ********

## Project Outline and Deadlines:

### Proposal (due: Mar 1 @ 11pm)

### Milestone (due: Mar 15, Mar 29, Apr 12, Apr 26 @ 11pm)

### Code (due: based on Milestones)

### Paper (due: Apr 24 @ 11pm)

### Peer-Reviews (due: May 1 @ 11pm)

### Demo (due: May 3 @ 1pm)

### Presentation (due: May 3 @ 1pm)
