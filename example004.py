import csv
import urllib
import numpy as np

def make_url(ticker_symbol):
    return "http://ichart.finance.yahoo.com/table.csv?s=" + ticker_symbol

#def make_filename(ticker_symbol, directory="examples"):
#    return "/home/mattliston/randomwalk/" + directory + "/" + ticker_symbol + ".csv"

def pull_data(ticker_symbol):
    try:
        file_handle = urllib.urlretrieve(make_url(ticker_symbol), ticker_symbol + ".csv")
#        print file_handle.readline()
    except urllib.ContentTooShortError as e:
#        with open(file_handle) as csvfile:
#            s_data = csv.reader(csvfile, delimiter = ',')
#            return s_data
         pass
#def __unicode__(self):
#   return unicode(self.some_field) or u'' #useless

tickers = [] #Get list of tickers & countries
countries = []
with open('data.csv') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    for row in data:
#        print row
        tickers.append(row[0])
        countries.append(row[3])
print tickers
print len(tickers)
#print countries

unique_countries = [] #Count unique countries for fun
for i in range(4, len(countries)): #pos 4 where white space ends
    if countries[i] not in unique_countries:
        unique_countries.append(countries[i])
    else:
        pass
print unique_countries
print len(unique_countries)

dates = []
open_prices = []
close_prices = []
high_prices = []
low_prices = []
volumes = []
adjusted_close = []
print tickers[4]
for s in range(4, 1000): 
    op = [] # open price
    cp = [] # close price
    hp = [] # high price
    lp = [] # low price
    v = [] # volume
    d = [] # dates
    adj = [] #adjusted close
#    if pull_data(url) is None:
#        continue
#    else:
    pull_data(tickers[s]) #create local csv file
    with open(tickers[s] + ".csv") as csvfile:
        s_data = csv.reader(csvfile, delimiter = ',')
#        print row[0]
#        if row[0] == "Date":
#        print row[1] #never reaches
#        print s_data
        try:
            for row in s_data:
#            print row
#                print tickers[s]
#                print row[1]
                d.append(row[0]) #this isn't working properly
                op.append(row[1])
                hp.append(row[2])
                lp.append(row[3])
                cp.append(row[4])
                v.append(row[5])
                adj.append(row[6])
        except IndexError:
            continue
#            print op
    dates.append(d)
    open_prices.append(op)
    high_prices.append(hp)
    low_prices.append(lp)
    volumes.append(v)
    adjusted_close.append(adj)


print open_prices #uncomment here
print len(open_prices)
#for i in range(1, len(tickers)):
    

#pull_data(make_url('PSAN.DE'))


























#import pandas as pd

#foo = open('companylist.csv')
#r_data = foo.read().splitlines()

#f_data = []
#for i in range(0, len(r_data)):
#    f_data.append(r_data.split(','))

#print f_data















