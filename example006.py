import urllib2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
print 'numpy ' + np.__version__
import random 
from scipy import stats
import argparse
import datetime

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
act=[]
date=[]
for i in range(1, ndays):
    # Date,Open,High,Low,Close,Volume,Adj Close
    spy_r.append(float(spy[i].split(',')[4]) / float(spy[i+args.horizon].split(',')[4]) - 1)
    act.append(float(spy[i].split(',')[4]))
    date.append(datetime.datetime.strptime(spy[i].split(',')[0], "%Y-%m-%d").date())

x = np.array(spy_r, dtype='float')
print 'x.shape', x.shape
y = np.random.normal(np.mean(x), np.std(x), 100000)
print 'y.shape', y.shape
z = np.array(act, dtype='float')
print 'z.shape', z.shape

fig, ax = plt.subplots(2, 1, sharex=False, sharey=False)

ax[0].hist(y, 100, normed=1, facecolor='green', alpha=0.75)
ax[0].hist(x, 100, normed=1, facecolor='red', alpha=0.50)
ax[0].set_title(vars(args))

ax[1].fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax[1].plot(date,z,color='k')
ax2 = ax[1].twinx()
ax2.plot(date,x,color='r')

plt.show()
