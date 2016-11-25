import csv
import urllib

def make_url(ticker_symbol):
    return "http://ichart.finance.yahoo.com/table.csv?s=" + ticker_symbol

def pull_data(url):
    try:
        file_handle = urllib.urlopen(url)
        print file_handle.readline()
    except urllib.ContentTooShortError as e:
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

#for s in range(4, len(tickers)): #this only reads the first line of the data from yahoo if you uncomment it... not sure why
#    op = [] # open price
#    cp = [] # close price
#    hp = [] # high price
#    lp = [] # low price
#    v = [] # volume
#    d = [] # dates
#    adj = [] #adjusted close
#    url = make_url(tickers[s])
#    print url
#    if pull_data(url) is None:
#        continue
#    else:
#        read = pull_data(url)
#    print read
#    with open(read) as csvfile:
#        s_data = csv.reader(csvfile, delimiter = ',') 
#        print s_data
#        for row in s_data:
#            print row
#            d.append(row[0]) #not necessary
#            op.append(row[1])
#            hp.append(row[2])
#            lp.append(row[3])
#            cp.append(row[4])
#            v.append(row[5])
#            adj.append(row[6])
#        print op
#    dates.append(d)
#    open_prices.append(op)
#    high_prices.append(hp)
#    low_prices.append(lp)
#    volumes.append(v)
#    adjusted_close.append(adj)


#print high_prices #uncomment here
#for i in range(1, len(tickers)):
    

#pull_data(make_url('PSAN.DE'))


























#import pandas as pd

#foo = open('companylist.csv')
#r_data = foo.read().splitlines()

#f_data = []
#for i in range(0, len(r_data)):
#    f_data.append(r_data.split(','))

#print f_data















