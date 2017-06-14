# CUDA_VISIBLE_DEVICES='0' python gan.py
import argparse
import struct
import time
import numpy as np
print 'numpy ' + np.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print 'tensorflow ' + tf.__version__
import cv2
print 'cv2 ' + cv2.__version__

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', help='latent space dimensionality', default=10, type=int)
parser.add_argument('--n', help='number of units per layer', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

with open('train-images-idx3-ubyte','rb') as f:
    h = struct.unpack('>IIII',f.read(16))
    d = np.fromstring(f.read(), dtype=np.uint8).reshape((h[1],h[2],h[3],1)).astype('float32')
    d = d/255. - .5

print 'd.shape',d.shape, 'd.min()',d.min(),'d.max()',d.max()

def dnet(args,x,reuse=None):
    print 'discriminator network, reuse',reuse
    with tf.variable_scope('dnet',reuse=reuse):
        d = tf.layers.conv2d(inputs=x, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print d
        d = tf.layers.conv2d(inputs=d, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print d
        d = tf.image.resize_bilinear(images=d,size=[14,14]) ; print d
        d = tf.layers.conv2d(inputs=d, filters=2*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print d
        d = tf.layers.conv2d(inputs=d, filters=2*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print d
        d = tf.image.resize_bilinear(images=d,size=[7,7]) ; print d
        d = tf.layers.conv2d(inputs=d, filters=3*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print d
        d = tf.layers.conv2d(inputs=d, filters=3*args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print d
        d = tf.contrib.layers.flatten(d)
        d = tf.layers.dense(inputs=d, units=1, activation=tf.sigmoid) ; print d
    return d

def gnet(args,z,reuse=None):
    print 'generator network, reuse', reuse
    with tf.variable_scope('gnet',reuse=reuse):
        g = tf.layers.dense(inputs=z, units=8*8*args.n, activation=None) ; print g
        g = tf.reshape(g,[-1,8,8,args.n]) ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.image.resize_bilinear(images=g,size=[14,14]) ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.image.resize_bilinear(images=g,size=[28,28]) ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.layers.conv2d(inputs=g, filters=args.n, kernel_size=3, strides=1,activation=tf.nn.elu, padding='same') ; print g
        g = tf.layers.conv2d(inputs=g, filters=1, kernel_size=3, strides=1,activation=None, padding='same') ; print g
    return g

x = tf.placeholder('float32', [None,28,28,1],name='x') ; print x
z = tf.placeholder('float32', [None,args.m],name='z') ; print z

dx = dnet(args,x) # d(x)
gz = gnet(args,z) # g(z)
dgz = dnet(args,gz,reuse=True) # d(g(z))

dxreal = tf.negative(tf.reduce_mean(tf.log(dx)))
dgzfake = tf.negative(tf.reduce_mean(tf.log(1-dgz)))
dgzreal = tf.negative(tf.reduce_mean(tf.log(dgz)))

dopt = tf.train.AdamOptimizer(learning_rate=args.lr)
dxreal_train = dopt.minimize(dxreal)
dgzfake_train = dopt.minimize(dgzfake)

gopt = tf.train.AdamOptimizer(learning_rate=args.lr)
dgzreal_train = gopt.minimize(dgzreal)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        np.random.shuffle(d)
        dxreal_loss=0.
        dgzfake_loss=0.
        dgzreal_loss=0.
        t=0.
        for j in range(0,d.shape[0],args.batch):
            _,dxreal_loss_ = sess.run([dxreal_train,dxreal],feed_dict={x:d[j:j+args.batch]})
            _,dgzfake_loss_ = sess.run([dgzfake_train,dgzfake],feed_dict={z:np.random.randn(args.batch,args.m)})
            _,dgzreal_loss_ = sess.run([dgzreal_train,dgzreal],feed_dict={z:np.random.randn(args.batch,args.m)})
            dxreal_loss += dxreal_loss_
            dgzfake_loss += dgzfake_loss_
            dgzreal_loss += dgzreal_loss_
            t+=1.
        print 'epoch',i,'dxreal',dxreal_loss/t,'dgzfake',dgzfake_loss/t,'dgzreal',dgzreal_loss/t
        x0 = sess.run(gz, feed_dict={z:np.random.randn(args.batch,args.m)})
        x0 = np.clip(x0+.5,0.,1.)*255.
        cv2.imshow('img', cv2.resize(np.concatenate((x0[0:10]).astype('uint8'),axis=1),(1000,100)))
        cv2.waitKey(10)
