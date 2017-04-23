import matplotlib.pyplot as plt
import cv2  
import os
import numpy as np
import mxnet as mx 
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle 
from sklearn.cross_validation import train_test_split
import logging
import dfk_cnn 

def get_args(arglist=None):
    parser = argparse.ArgumentParser(description='set up the input path of image files')
    parser.add_argument('--train', type=str, default='.',
                        help = 'the input path of imgs')
    parser.add_argument('--test', type=str, default='.',
                        help = 'the path of test imgs')
    parser.add_argument('--output', type=str, default='.',
                        help='the model ')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.01,
                        help='momentum')
    parser.add_argument('--wd', type=float, default=0.9,
                        help='wd')
    parser.add_argument('--num_epoch', type=int, default=50,
                        help='num_epoch')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu')
    parser.add_argument('--prefix', type=str, default='.',
                        help='model save path ')

    if arglist is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)

args=get_args()

FTRAIN = args.train+'/training.csv'
FTEST = args.train+'/test.csv' 

# Define the helper function for loading the data 
def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

#Load and check the data
X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max())) 

def to4d(img):
    return img.reshape(img.shape[0], 1, 96, 96).astype(np.float32) 
#load data
# X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
train = mx.io.NDArrayIter(data =to4d(X_train), label = y_train, batch_size = 128)
val = mx.io.NDArrayIter(data =to4d(X_test), label = y_test, batch_size = 128)
kv = mx.kvstore.create('local')
head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
#model

model = mx.model.FeedForward(
        ctx                = mx.cpu(), # change to mx.gpu() if GPU used 
        symbol             = fkeyCNN,
        num_epoch          = args.num_epoch,
        learning_rate      = args.learning_rate,
        momentum           = args.momentum,
        )
model.fit(
        X                  = train,
        eval_data          = val,
        batch_end_callback = mx.callback.Speedometer(1, 50),
        epoch_end_callback = None,
        eval_metric='rmse' ) 

#Save and load the model
prefix = args.prefix 
iteration = 50
model.save(prefix) 

