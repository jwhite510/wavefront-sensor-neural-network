# wavefront sensor neural network
neural network trained to retrieve the amplitude and phase of objects by measurement of their diffraction pattern

publication:
Real-time phase-retrieval and wavefront sensing enabled by an artificial neural network  
Jonathon White, Sici Wang, Wilhelm Eschen, and Jan Rothhardt  
https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-29-6-9283&id=449096  


# installation
```
pip install -r requirements.txt
```
# Overview / Files Explained in order of importance
### run\_tests.sh
To train the network, run the shellscript 'run\_tests.sh'. This will call several .py files to create a new data set for training, add artificial noise to the dataset, and then train a neural network. Parameters for training the network are specified in this file, these parameters are the number of training samples, the peak counts of the noise levels for the training data, and the wavefront sensor.

### noise\_test\_compareCDI.sh
Load the trained network and perform retrieval on several test samples. Also perform retrieval on the same samples using iterative phase retrieval for comparison


### diffraction\_net.py
Contains the neural network, and classes for training the neural network, run from file 'run\_tests.sh'

### datagen.py
Contains functions/classes for generating wavefronts and propagating them through the wavefront, used to create the training data set

### addnoise.py
Adds poisson noise to simulate low photon count / additive noise from camera.

### buildui.sh / main.py / \_main.py / main.ui
Files for building a user interface in Qt with file for use in realtime phase retrieval when connected to a camera.










