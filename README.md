# Team Pandemonium

<details>
<summary>Installing Dependencies</summary>
<br>

### Installing Dependences
Install the python package manager PIP  

```python<3> get-pip.py``` or ```sudo easy_install pip```  

__NOTE:__ Make sure to use python 3.

### Install virutalenv
install virtualenv using pip:  

```pip<3> install virtualenv```

virtualenv is used to create and manage environments for different python projects.  Use virtualenv to create a virtual environment by using:

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
</details>

<details>
<summary>Data Preprocessing</summary>
<br>

## Data preprocessing and preparation

### NLTK parts of speech data  

The NLTK library requires you to download corpora data for our part of speech tagging.  You can download and install this data via a python interpreter:

```
>>> import nltk  
>>> nltk.download()
```

A dialog window will pop-up after the function call which allows you to select and install data.

## Downloading cartoon script data  
Our project uses the American cartoon show, Rick and Morty, to create our training corpus. The script can be acquired by running ```python utils/preprocess.py``` from the root directory, which will download and format the data, placing it in `/data/train/cleaned/simple.txt`

</details>

<details>
<summary>Training and Generating</summary>
<br>

## Training and Testing 

### Train
`train.py` is where the model is located, and this is where you can tweak any hyperparameters you'd like. Once you have the model you desire, run ```python train.py```, which will train the model on any data that is located in `data/train/cleaned/simple.txt`.
This script also produces an image of the training curves, called `training_curves_pos.png` or `training_curves_no_pos.png`, depending on if part of speech was included as input or not.

### Generate Text 
After the network is trained, you can run ```python test.py```, which will load the model and generate 50 sentences. Similar to above, the output will be stored in `pos_output.txt` or `no_pos_output.txt`, depending on if part of speech was included as input or not.
</details>

<details>
<summary>Comparison Metrics</summary>
<br>

## Comparisons/Metrics

Now that you have generated some sentences, its time to do some comparisons to get some statistics.
### Statistics
Running ```python stats.py``` will calculate the total number of words in your corpus, the total number of unique words in your corpus, and then the outlier count (words that appear 5 times or less). These statistics will be written to stat.txt

### Markov Sentences
Running ```python markov.py``` will generate 50 markov sentences based on the text stored in `data/train/cleaned/simple.txt`.
The markov sentences are written to a file named `markovSentences.txt`.

### Metrics
metrics.py is a script that takes two .txt files as arguments and compares each file sentence by sentence, calculating the average hamming, cosine, Gotoh and Levenshtein distances between the sentences. It compares the part of speech tags of the words in each sentence, as opposed to the actual words. 

Use ```python metrics.py --file1 firstFilename.txt --file2 secondFilename.txt``` to compare the distances between the two files. It will output the distances as it runs, and then writes the statistics to `metricStats.txt`.
To compare the sentences generated by the markov model and the sentences generated by the neural network, run

```python metrics.py --file1 markovSentences.txt --file2 ../pos_output.txt```

</details>
