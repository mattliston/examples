import urllib2
import matplotlib.pyplot as plt
import numpy as np
print 'numpy ' + np.__version__
import random 
from scipy import stats
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ticker', help='ticker symbol', required=True)
parser.add_argument('--horizon', help='return horizon',default=250,  type=int)
args = parser.parse_args()
print(args)

spy = urllib2.urlopen('http://real-chart.finance.yahoo.com/table.csv?s='+args.ticker).read().splitlines()

ndays = len(spy) - args.horizon
print 'ndays', ndays

spy_r=[]
for i in range(1, ndays):
    # Date,Open,High,Low,Close,Volume,Adj Close
    spy_r.append(float(spy[i].split(',')[4]) / float(spy[i+args.horizon].split(',')[4]) - 1)

x = np.array(spy_r, dtype='float')
print 'x.shape', x.shape

y = np.random.normal(np.mean(x), np.std(x), 100000)
print 'y.shape', y.shape

plt.hist(y, 100, normed=1, facecolor='green', alpha=0.75)
plt.hist(x, 100, normed=1, facecolor='red', alpha=0.50)
plt.title(vars(args))
plt.show()
