import mxnet as mx  

def distDrCNN():
    data    = mx.symbol.Variable('data')
    # first convolution
    conv1   = mx.symbol.Convolution(data=data, kernel=(8,6), num_filter=20)
    act1    = mx.symbol.Activation(data=conv1, act_type='tanh')
    pool1   = mx.symbol.Pooling(data=act1, pool_type='max',
                                kernel=(2,2), stride=(2,2))

    # second convolution
    conv2   = mx.symbol.Convolution(data=pool1, kernel=(6,4), num_filter=64)
    act2    = mx.symbol.Activation(data=conv2, act_type='tanh')
    pool2   = mx.symbol.Pooling(data=act2, pool_type='max',
                                kernel=(2,2), stride=(2,2))

    # third convolution
    conv3   = mx.symbol.Convolution(data=pool2, kernel=(4,3), num_filter=90)
    act3    = mx.symbol.Activation(data=conv3, act_type='tanh')
    pool3   = mx.symbol.Pooling(data=act3, pool_type='max',
                                kernel=(2,2), stride=(2,2))
    # first fully connected
    flatten = mx.symbol.Flatten(data=pool3)
    hidden4 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    act4    = mx.symbol.Activation(data=hidden4,act_type='tanh')

    # second fully connected
    hidden5 = mx.symbol.FullyConnected(data=act4, num_hidden=200)
    act5    = mx.symbol.Activation(data=hidden5, act_type='tanh')

    # output layer
    output  = mx.symbol.FullyConnected(data=act5, num_hidden=10)
    
    # soft loss function
    lenet = mx.sym.SoftmaxOutput(data=output,name='softmax')
    return lenet

if __name__=="__main__":
   dCNNinst=distDrCNN()
   batch_size = 200
   BSIZE=160
   HSIZE=120
   shape = {"data" : (batch_size, 1, BSIZE, HSIZE)}   
   mx.viz.plot_network(symbol=dCNNinst, shape=shape)

