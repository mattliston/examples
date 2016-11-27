import csv
import urllib
import numpy as np

def create_data():

    def pull_data(ticker_symbol):
        try:
            file_handle = urllib.urlretrieve("http://ichart.finance.yahoo.com/table.csv?s=" + ticker_symbol, ticker_symbol + ".csv")
        except urllib.ContentTooShortError as e:
            pass

    tickers = [] #Get list of tickers & countries
    countries = []
    with open('data.csv') as csvfile:
        data = csv.reader(csvfile, delimiter = ',')
        for row in data:
            tickers.append(row[0])
            countries.append(row[3])

    unique_countries = [] #Count unique countries for fun
    for i in range(4, len(countries)): #pos 4 where white space ends
        if countries[i] not in unique_countries:
            unique_countries.append(countries[i])
        else:
            pass

    dates = []
    open_prices = []
    close_prices = []
    high_prices = []
    low_prices = []
    volumes = []
    adjusted_close = []
   
    for s in range(4, 20): 
        op = [] # open price
        cp = [] # close price
        hp = [] # high price
        lp = [] # low price
        v = [] # volume
        d = [] # dates
        adj = [] #adjusted close
        pull_data(tickers[s]) #create local csv file
        with open(tickers[s] + ".csv") as csvfile:
            s_data = csv.reader(csvfile, delimiter = ',')
            try:
                for row in s_data:
                    try:
                        d.append((row[0])) #this isn't working properly
                        op.append(float(row[1]))
                        hp.append(float(row[2]))
                        lp.append(float(row[3]))
                        cp.append(float(row[4]))
                        v.append(float(row[5]))
                        adj.append(float(row[6]))
                    except ValueError:
                        continue
            except IndexError:
                continue

        dates.append(d)
        open_prices.append(op)
        high_prices.append(hp)
        low_prices.append(lp)
        volumes.append(v)
        adjusted_close.append(adj)

        np_dates = np.array(dates)
        np_open_prices = np.array(open_prices)
        np_high_prices = np.array(high_prices)
        np_low_prices = np.array(low_prices)
        np_volumes = np.array(volumes)
        np_adjusted_close = np.array(adjusted_close)
        return np_dates, np_open_prices, np_high_prices, np_low_prices, np_volumes, np_adjusted_close

print create_data()       
#d = Data()
#print d.create_data()
#overvalued, undervalued
#(sign, magnitude)









































