# wget http://stuff.mit.edu/afs/sipb/contrib/pi/pi-billion.txt
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python example009.py
from __future__ import division
import numpy as np
import theano
import theano.tensor as T
import lasagne as L
import argparse
import time
from six.moves import cPickle
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=200)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

print 'numpy ' + np.__version__
print 'theano ' + theano.__version__
print 'lasagne ' + L.__version__

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ndigits', help='number of digits, default 1000000', default=1000000, type=int)
parser.add_argument('--window', help='window size, default=100', default=100, type=int)
parser.add_argument('--lr', help='learning rate, default 0.001', default=0.001, type=float)
parser.add_argument('--nepoch', help='number of epochs, default=100', default=100, type=int)
parser.add_argument('--nbatch', help='number of batches per eopch, default=100', default=100, type=int)
parser.add_argument('--batchsize', help='batch size, default 1000', default=1000, type=int)
parser.add_argument('--test', help='test fraction, default 0.2', default=0.2, type=float)
parser.add_argument('--model', help='output model filename')
args = parser.parse_args()
print args

# load data
with open('pi-billion.txt') as f:
    s = f.read()
f.close()

pi = np.empty([args.ndigits],dtype='float32')
i=0
for c in s:
    if c.isdigit():
        pi[i] = float(c)
        i+=1
        if i==args.ndigits:
            break
print 'pi.shape',pi.shape

input_var = T.matrix(dtype=theano.config.floatX)
target_var = T.vector(dtype='int32')
network = L.layers.InputLayer((None, args.window), input_var)
print 'input', L.layers.get_output_shape(network)
network = L.layers.ReshapeLayer(network, ((-1, 1, args.window)))
print 'reshape', L.layers.get_output_shape(network)
network = L.layers.Conv1DLayer(network,num_filters=256,filter_size=11,stride=2)
print 'conv', L.layers.get_output_shape(network)
network = L.layers.Conv1DLayer(network,num_filters=256,filter_size=11,stride=2)
print 'conv', L.layers.get_output_shape(network)
network = L.layers.Conv1DLayer(network,num_filters=256,filter_size=11,stride=2)
print 'conv', L.layers.get_output_shape(network)
network = L.layers.Conv1DLayer(network,num_filters=256,filter_size=11,stride=2)
print 'conv', L.layers.get_output_shape(network)
conv = L.layers.Conv1DLayer(network,num_filters=256,filter_size=11,stride=2)
print 'conv', L.layers.get_output_shape(conv)
gap = L.layers.Pool1DLayer(conv, pool_size=L.layers.get_output_shape(conv)[2], stride=None, pad=0, mode='average_inc_pad')
print 'gap', L.layers.get_output_shape(gap)
network = L.layers.DenseLayer(gap, 2, nonlinearity=L.nonlinearities.softmax)
print 'output', L.layers.get_output_shape(network)

#input_var = T.matrix(dtype=theano.config.floatX)
#target_var = T.vector(dtype='int32')
#network = L.layers.InputLayer((None, args.window), input_var)
#network = L.layers.DenseLayer(network, 10000)
#network = L.layers.DenseLayer(network, 1000)
#network = L.layers.DenseLayer(network, 1000)
#network = L.layers.DenseLayer(network, 1000)
#network = L.layers.DenseLayer(network, 1000)
#network = L.layers.DenseLayer(network, 1000)
#network = L.layers.DenseLayer(network, 100)
#network = L.layers.DenseLayer(network, 2, nonlinearity=L.nonlinearities.softmax)

prediction = L.layers.get_output(network)
loss = L.objectives.aggregate(L.objectives.categorical_crossentropy(prediction, target_var), mode='mean')
params = L.layers.get_all_params(network, trainable=True)
updates = L.updates.adam(loss, params, learning_rate=args.lr)
scaled_grads,norm = L.updates.total_norm_constraint(T.grad(loss,params), np.inf, return_norm=True)
train_fn = theano.function([input_var, target_var], [loss,norm], updates=updates)
test_fn = theano.function([input_var], L.layers.get_output(network, deterministic=True))

d = np.empty([args.batchsize,args.window],dtype='float32')
l = np.empty([args.batchsize],dtype='int32')
t0 = time.time()
t = time.time()
for i in range(args.nepoch):
    tloss=0
    tnorm=0
    #train
    for j in range(args.nbatch):
        for k in range(args.batchsize):
            #w = np.random.randint(int(pi.shape[0]*args.test),pi.shape[0]-args.window)
            w = np.random.randint(0,int(pi.shape[0]*(1-args.test))-args.window)
            d[k] = pi[w:w+args.window]
            if np.random.randint(0,2)==0:
                l[k]=0
            else:
                np.random.shuffle(d[k])
                l[k]=1
        bloss,bnorm = train_fn(d,l)
        tloss += bloss
        tnorm += bnorm
    #test 
    for k in range(args.batchsize):
        #w = np.random.randint(0,int(pi.shape[0]*args.test-args.window))
        w = np.random.randint(int(pi.shape[0]*(1-args.test)),pi.shape[0]-args.window)
        d[k] = pi[w:w+args.window]
        if np.random.randint(0,2)==0:
            l[k]=0
        else:
            np.random.shuffle(d[k])
            l[k]=1
    val_output = test_fn(d)
    val_predictions = np.argmax(val_output, axis=1)
    tacc = np.mean(val_predictions == l)

    print 'epoch {:8d} loss {:12.8f} grad {:12.8f} accuracy {:12.8f} n_zero {:6d} n_one {:6d} t_epoch {:4d} t_total {:8d}'.format(i, tloss/args.nbatch, tnorm/args.nbatch, tacc, np.sum(val_predictions==0), np.sum(val_predictions==1), int(time.time()-t), int(time.time()-t0))
    t = time.time()

f = open(args.model, 'wb')
cPickle.dump(L.layers.get_all_param_values(network), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
