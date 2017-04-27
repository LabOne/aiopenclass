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
