import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print 'tensorflow ' + tf.__version__

import numpy as np
print 'numpy ' + np.__version__

import cv2
print 'cv2 ' + cv2.__version__

import igraph
print 'igraph ' + igraph.__version__

import scipy
print 'scipy ' + scipy.__version__
