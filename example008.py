import urllib2
import random
import numpy as np
import scipy.special as special
import scipy.stats as stats
import matplotlib.pyplot as plt
#def gauss(x):
#    return np.random.normal(np.mean(x),np.std(x),100000)

def generate_labels(obs):
    z_scores = stats.zscore(obs)
    return 1-special.ndtr(z_scores)

def plot_indv():
    spy = urllib2.urlopen('http://real-chart.finance.yahoo.com/table.csv?s=SPY').read().splitlines()
    ndays = len(spy) - 30
    print 'ndays', ndays
    spy_r=[]
    act=[]
#    date=[]
    for i in range(1, ndays):
        # Date,Open,High,Low,Close,Volume,Adj Close
        spy_r.append(float(spy[i].split(',')[4]) / float(spy[i+30].split(',')[4]) - 1)
        act.append(float(spy[i].split(',')[4]))
#        date.append(datetime.datetime.strptime(spy[i].split(',')[0], "%Y-%m-%d").date())
    spy_label = generate_labels(spy_r)
    print spy_label

    x = np.array(spy_r,dtype='float')
#Potential Labels
    y = np.array(spy_label,dtype='float')
    z = np.array(stats.norm.sf(stats.zscore(spy_r)),dtype='float') #Actually identical

    fig, ax = plt.subplots(3, 1, sharex=False, sharey=False)
    ax[0].hist(x, 1000, normed=1, facecolor='green', alpha=0.75)
    ax[0].set_title('Data')
    ax[1].hist(y, 1000, normed=1, facecolor='red', alpha=0.50)
    ax[1].set_title('Label1')
    ax[2].hist(z, 1000, normed=1, facecolor='blue', alpha=0.5)
    ax[2].set_title('Label2')
    plt.show()
plot_indv()

def windows():
    spy = urllib2.urlopen('http://real-chart.finance.yahoo.com/table.csv?s=SPY').read().splitlines()
    ndays = len(spy) - 30
    print 'ndays', ndays
    spy_r=[]
    act=[]
#    date=[]
    for i in range(1, ndays):
        # Date,Open,High,Low,Close,Volume,Adj Close
        spy_r.append(float(spy[i].split(',')[4]) / float(spy[i+30].split(',')[4]) - 1)
        act.append(float(spy[i].split(',')[4]))
#        date.append(datetime.datetime.strptime(spy[i].split(',')[0], "%Y-%m-%d").date())
    spy_label = generate_labels(spy_r)
    print spy_label

    x = np.array(spy_r,dtype='float')
#Potential Labels
    y = np.array(spy_label,dtype='float')

    window_length = 30
    window_labels = []
    for i in range(0,ndays-1):
        z = np.array(y[i:i+30])
        window_labels.append(stats.combine_pvalues(z)[1])
    print window_labels

#    fig, ax = plt.subplots(2, 1, sharex=False, sharey=False)
    plt.hist(window_labels, 1000, normed=1, facecolor='green', alpha=0.75)
    plt.title('Window Labels')
    plt.show()
windows()
