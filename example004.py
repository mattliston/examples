import csv
import urllib
import numpy as np
# add some formatting defaults for print
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=200)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})


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
                        d.append((row[0])) 
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
        print 'ticker',tickers[s],'len(adjusted_close)',len(adjusted_close),'len(adj)',len(adj),max(len(x) for x in adjusted_close)

#        np_dates = np.array(dates)
#        np_open_prices = np.array(open_prices) #ignore for now
#        np_high_prices = np.array(high_prices)
#        np_low_prices = np.array(low_prices)
#        np_volumes = np.array(volumes)#
    np_adjusted_close = np.empty([len(adjusted_close),max(len(x) for x in adjusted_close)], dtype='float') # numpy arrays are fixed size, create space for [num_tickers,largest_ticker_length]
    np_adjusted_close.fill(np.nan) # fill with default value = "not a number" (np.nan)
    for i in range(len(adjusted_close)):
        np_adjusted_close[i,0:len(adjusted_close[i])] = adjusted_close[i]
    print 'np_adjusted_close.shape', np_adjusted_close.shape
    np_c = np.fliplr(np_adjusted_close) # make oldest values first (flip each row)
    print np_c[:,-30:]
    monthly_change = np_c[:,30:] / np_c[:,0:-30]
    print 'monthly_change.shape', monthly_change.shape
    print monthly_change[:,-10:] # print last 10 entries of each ticker (negative indices count from the end)
    exit(0)

    np_adjusted_close = np.array([np.array(adjusted_closei) for adjusted_closei in adjusted_close]) #use
    np.flipud(np_adjusted_close) #make oldest values first
    print np_adjusted_close[5][1]
    print np_adjusted_close[1][30]/np_adjusted_close[1][0]
    monthly_change = [np_adjusted_close[i][30:]/np_adjusted_close[i][0:] for i in np_adjusted_close] #this line
#        print np_adjusted_close[0][1]
#    print np_adjusted_close
#    print np_adjusted_close.size
#    print type(np_adjusted_close[1])
#    print np_adjusted_close[0:,30:]/np_adjusted_close[0::]     
#    for i in range(0, np_adjusted_close.size):
#        print np_adjusted_close[0][i] 
#    total = []
#    input_vector = [] #this will be list of 3D lists
#        mean = np_adjusted_close[30:][0]/np_adjusted_close[0:][0]
#        print mean
#        for i in range(0, np_adjusted_close.size):
#            s_vector = [] #input vector for each stock
#            for j in range(0, np_adjusted_close.size):
#                try:
#                    change = (np_adjusted_close[0][i+30]/np_adjusted_close[0][i]) - 1
#                    print change
#                    total.append(change)
#                    sign = 0
#                    if change > 0:
#                       sign = 1
#                    elif change == 0:
#                       sign = 0
#                    else: 
#                       sign = -1
                    
#write code to find magnitude
    # find average return of all stocks
    # average return stock(i) - average return(market) = expected sliding monthly return E(R)
    # center distribution at E(R)
    # calculate probability of x1 occurring given distribution
    # condense to single order of magnitude 
#                except IndexError:
#                    break
#        print total
#    np_total = np.array(total)
#    mean_total = []
#    for i in range(0, np_total.size):
#        mean_total = np.mean(np_total[i])
#            for j in range(0, np_total[i].size)
#        print mean_total        
#        mean_total_return = np.mean(np_total)
#       mean_sreturn = []
#        for i in range(0, mean_total_return.size):
#            mean_sreturn.append(np.mean(mean_total_return[i] - mean_total_return))
#        print mean_sreturn
#conversion to tensor theano.tensor.as_tensor_variable(x, name=None, ndim=None)
        
#30 day moving average

#np array train data & labels
#np array test data & labels
#        return np_dates, np_open_prices, np_high_prices, np_low_prices, np_volumes, np_adjusted_close

create_data()       
#d = Data()
#print d.create_data()
#overvalued, undervalued
#(sign, magnitude)









































