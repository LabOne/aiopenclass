# Dependency Installation Instructions
This repository requires the following dependencies installable via pip:
*    pandas
*    scikit-learn
*    scikit-image
*    opencv-python
*    keras
*    h5py
*    flask
*    keras
*    nose

**Note**: We use a special tensorflow library from Google Deepmind called sonnet[https://github.com/deepmind/sonnet]. Find the installation instructions at the github repository which we have linked.

We will be providing compressed docker images during the class vias USB due to the slow internet speed, and have uploaded the docker image to (baidu cloud)[http://pan.baidu.com/s/1deDfH6p]. (*Note*: if you already downloaded the docker image from (docker hub)[https://hub.docker.com/r/raulpuric/tf_gmic/] it is no longer current)

You can load the compressed docker image from the USB into docker via `docker load --input gmic_tf_ws.tar`.

# Demo preparation
After cloning this repository to a location on your computer `<path name>`.
make the following directory: `mkdir <path name>/tensorflow_workshop/data/`
Copy the necessary data zip from the provided (baidu cloud link)[https://pan.baidu.com/s/1geNQYtl] with passowrd x5yr to the directory `<path name>/tensorflow_workshop/data/<data file name>`.

Change directory with `cd <path name>/tensorflow_workshop/`
Use the docker image we just loaded to instantiate a container and mount our github repository+code `docker run -it -p 8888:8888 -v "$(pwd)":/root/tensorflow_workshop <loaded docker image name>`.

Run the environment prep script with `sh prep_env.sh`.

# Files in this repo

### basic.py
Contains basic example to instantiate and run a Tensorflow graph.
  
### board.py
Contains basic example to visualize a Tensorflow graph with TensorBoard.

### capt_gen.py
Example to train (and test) a caption generator from images with a pretrained VGG-16 image feature extractor.

### flask-captgen.py
Example to run a flask webapp to caption images.

### neural-keras.py
Train an mnist classifier with the keras high-level API.

### neural.py
Train an mnist classifier with native tensorflow.

### post_capt_gen(does not work).py
Failed example trying to use variable scope to reload tensorflow LSTM weights at test time.

### post_capt_gen.py
Example using hacky variable scoping to reload tensorflow LSTM weights at test time. 

### pre_sonnet_capt_gen(does not work).py
Second attemmpt to use variable scoping to reload tensorflow LSTM weights.

### pre_sonnet_capt_gen.py
Third attempt to use variable scoping to reload tensorflow LSTM weights (still doesn't work).

### sonnet_capt_gen.py
Example to use sonnet to perform object oriented approach to LSTM variable reloading.

### vae_seq2seq.py
Using Variational Autoencoder to perform unsupervised learning of text sequences.

### model_serve.py + main.cc
Walkthrough of how to serve tensorflow model on a gRPC server endpoint. Also includes example of how to use tensorflow api for protobuf serialization

### train_distributed.py
Walkthrough of how to train a blackbox neural net model across multiple GPUs with data parallelism.
