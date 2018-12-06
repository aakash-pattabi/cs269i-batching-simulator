import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from multiprocessing import Process, Manager
import os

## Order directions
SELL = -1
BUY = 1

## Events
NEWORDER = 1
CANCEL = 2
DELETE = 3
EXECUTE_VISIBLE = 4
EXECUTE_HIDDEN = 5

## Current tracking variables
CUR_ASK = 0
CUR_BID = 0

## Current order book; initialize to empty
BOOK = None

## Cumulative tracking variables for batching simulator
BATCHES = 0
VOLUME_CLEARED_BATCH = 0
TRADES_CLEARED_BATCH = 0

CUM_VOLUME_BATCH = []
CUM_TRADES_BATCH = []
PRICE_BATCH = []
BOOK_SIZE_BATCH = []
BATCH_INTERVAL = []

## Cumulative tracking variables for experience re-run of 
## observed data
CUM_VOLUME_EXP = []
CUM_TRADES_EXP = []
PRICE_EXP = []

###################################################################################################
###################################################################################################

'''
Method: computeEquilibrium
Arguments: p_s = vector of prices in sell orders, 
		   p_d = vector of prices in buy orders, 
		   q_s = vector of quantities in sell orders, 
		   q_d = vector of quantities in buy orders
Returns: clearing = market clearing price for batch

Description: 

'''
def computeEquilibrium(p_s, p_d, q_s, q_d):
	dup = set(p_s).intersection(set(p_d))
	clearing = None
	try:
		clearing = np.amin(list(dup))
	except ValueError:
		aggs = np.array([p_s, q_s]).T
		aggd = np.array([p_d, q_d]).T 
		## Conduct forward search over space of equilibria: 
		for obs in aggs:
			if 0 in np.sum(np.less(obs, aggd), axis = 1):
				## What weird syntax for np.where... 
				idx = np.where(np.all(aggs == obs, axis = 1))[0][0]
				clearing = np.average([aggs[idx-1, 0], aggs[idx, 0]])
				break
	return clearing

'''
Method: processBook
Arguments: orderBook = rectangular array of current order book
Returns: orderBook = updated order book

Description: 

'''
def processBook(orderBook):
	global BATCHES, VOLUME_CLEARED_BATCH, TRADES_CLEARED_BATCH, PRICE_BATCH, \
		BOOK_SIZE_BATCH, CUM_VOLUME_BATCH, CUM_TRADES_BATCH

	## Process CANCELs and DELETEs before computing equilibrium price and
	## quantity traded
	orders_to_delete = orderBook[orderBook[:,1] == DELETE, 2]
	if len(orders_to_delete) > 0:
		orderBook = orderBook[~np.isin(orderBook[:,2], orders_to_delete),:]

	orders_to_cancel = orderBook[orderBook[:,1] == CANCEL,:]
	orderBook = orderBook[orderBook[:,1] != CANCEL,:]
	if orders_to_cancel.shape[0] > 0:
		cum_cancellations = pd.DataFrame(orders_to_cancel[:,2:4]).groupby(0, as_index = False).sum()
		cum_cancellations = np.array(cum_cancellations)

		for cancellation in cum_cancellations:
			orderBook[(orderBook[:,2] == cancellation[0]),3] -= cancellation[1]

	## Bid-ask spread is the difference between the best (highest) bid
	## demand price and the best (lowest) ask supply price

	## Current ask is the minimum price in the order book for which
	## Direction == SELL
	CUR_ASK = sys.maxsize
	try:
		CUR_ASK = np.amin(orderBook[orderBook[:,5] == SELL, 4])
	except ValueError:
		pass

	## Current bid is the maximum price in the order book for which 
	## Direction == BUY 
	CUR_BID = 0
	try:
		CUR_BID = np.amax(orderBook[orderBook[:,5] == BUY, 4])
	except ValueError:
		pass

	if CUR_BID < CUR_ASK:
		## Price stays the same in this batch period
		if len(PRICE_BATCH) == 0:
			PRICE_BATCH.append(0)
		else:
			PRICE_BATCH.append(PRICE_BATCH[-1])
		pass
	else:
		## Process trades
		buys = orderBook[orderBook[:,5] == BUY,:]
		buys = buys[buys[:,4].argsort()[::-1]]
		p_d = buys[:,4]
		q_d = np.cumsum(buys[:,3])

		sells = orderBook[orderBook[:,5] == SELL,:]
		sells = sells[sells[:,4].argsort()]
		p_s = sells[:,4]
		q_s = np.cumsum(sells[:,3])

		## All trades to the left of the market clearing price / 
		## quantity are executed to completion
		clearing = computeEquilibrium(p_s, p_d, q_s, q_d)
		PRICE_BATCH.append(clearing)
		VOLUME_CLEARED_BATCH += np.sum(orderBook[orderBook[:,4] < clearing, 3])
		TRADES_CLEARED_BATCH += orderBook[orderBook[:,4] < clearing,:].shape[0]
		orderBook = orderBook[orderBook[:,4] >= clearing,:]

		## Trades at the market clearing price / quantity are executed
		## at random until the thinner side of the market evaporates
		residual_demand = orderBook[(orderBook[:,4] == clearing) & (orderBook[:,5] == BUY)]
		residual_supply = orderBook[(orderBook[:,4] == clearing) & (orderBook[:,5] == SELL)]

		if residual_demand.shape[0] > 0 and residual_supply.shape[0] > 0:
			np.random.shuffle(residual_supply); np.random.shuffle(residual_demand)

			excess_supply = np.sum(residual_demand[:,3]) < np.sum(residual_supply[:,3])
			if excess_supply:
				## This is definitely consfusing... but the random clearing of trades at
				## the market clearing price is invariant to how the trades are labelled, 
				## so really this just saves lines
				tmp = residual_demand
				residual_demand = residual_supply
				residual_supply = tmp

			cum_residual_supply = np.sum(residual_supply[:,3])
			cum_residual_demand = np.cumsum(residual_demand[:,3])

			first_residual_incomplete = np.argmin(cum_residual_demand[cum_residual_demand >= cum_residual_supply])
			partial_residual_supply = cum_residual_supply - np.sum(residual_demand[:first_residual_incomplete,3])
			demand_residuals_left = residual_demand[first_residual_incomplete:,:]
			demand_residuals_left[0,3] -= partial_residual_supply

			VOLUME_CLEARED_BATCH += cum_residual_supply
			TRADES_CLEARED_BATCH += first_residual_incomplete + residual_supply.shape[0]

			orderBook = orderBook[orderBook[:,4] > clearing,:]
			orderBook = np.vstack((demand_residuals_left, orderBook))

	## Divide by two to prevent double-counting of trades and volume cleared
	CUM_VOLUME_BATCH.append(VOLUME_CLEARED_BATCH/2)
	CUM_TRADES_BATCH.append(TRADES_CLEARED_BATCH/2)
	BATCHES += 1
	BOOK_SIZE_BATCH.append(orderBook.shape[0])
	return orderBook

'''
Method: getToProcess
Arguments: orders = subset of data frame of daily orders, 
		   time = boolean if measuring stride in seconds
Returns: index of one order past the last one in the batch

Description: 

'''
def getToProcess(orders, time):
	orders = np.array(orders)
	max_time = orders[0, 0] + time
	last_order_in_batch = np.argmax(orders[orders[:,0] <= max_time,0])
	## Return + 1 so that ':' indexing includes the last order
	return (last_order_in_batch + 1)

'''
Method: simFrequentBatchAuction
Arguments: d = dictionary of results, 
		   orders = data frame of daily orders, 
		   stride = window of orders of seconds, 
		   time = boolean if measuring stride in seconds
Returns: None

Description: 

'''
def simFrequentBatchAuction(d, orders, stride, time = False):
	global BOOK, BATCHES, VOLUME_CLEARED_BATCH, TRADES_CLEARED_BATCH, \
		PRICE_BATCH, BOOK_SIZE_BATCH, BATCH_INTERVAL

	orders = orders.loc[orders['Event'].isin([NEWORDER, DELETE, CANCEL])]
	orders.reset_index(drop = True, inplace = True)
	while (orders.shape[0] > 0):
		idx = getToProcess(orders, stride) if time else stride
		to_process = np.array(orders.iloc[0:idx])
		interval = max(to_process[:,0]) - min(to_process[:,0])
		BATCH_INTERVAL.append(interval)
		orders.reset_index(drop = True, inplace = True)
		orders = orders.drop(range(0, min(idx, orders.shape[0])))
		BOOK = to_process if BOOK is None else np.vstack((BOOK, to_process))
		BOOK = processBook(BOOK)
		print ("Remaining: %d" % (orders.shape[0]))
		
	d["BATCH_INTERVAL"] = BATCH_INTERVAL
	d["CUM_VOLUME_BATCH"] = CUM_VOLUME_BATCH
	d["CUM_TRADES_BATCH"] = CUM_TRADES_BATCH
	d["PRICE_BATCH"] = PRICE_BATCH
	d["BOOK_SIZE_BATCH"] = BOOK_SIZE_BATCH
	d["BATCHES"] = BATCHES

'''
Method: processHistoricalOrders
Arguments: orders = subset of data frame of daily orders
Returns: None

Description: 

'''
def processHistoricalOrders(orders):
	global CUM_TRADES_EXP, CUM_VOLUME_EXP, PRICE_EXP

	orders = orders[orders[:,1] == EXECUTE_VISIBLE,:]
	try:
		## Divide by two to prevent double-counting of trades and volume cleared
		CUM_TRADES_EXP.append(orders.shape[0]/2)
		CUM_VOLUME_EXP.append(np.sum(orders[:,3])/2)
		## Periodic price is last traded price in period
		PRICE_EXP.append(orders[-1,4])
	except IndexError:
		if len(PRICE_EXP) == 0:
			PRICE_EXP.append(0)
		else:
			PRICE_EXP.append(PRICE_EXP[-1])

'''
Method: reconstructOrderBook
Arguments: d = dictionary of results, 
		   orders = data frame of daily orders, 
		   stride = window of orders of seconds, 
		   time = boolean if measuring stride in seconds
Returns: None

Description: 

'''	
def reconstructOrderBook(d, orders, stride, time = False):
	global CUM_VOLUME_EXP, CUM_TRADES_EXP, VOLUME_CLEARED_EXP, TRADES_CLEARED_EXP, PRICE_EXP

	orders = orders.loc[orders['Event'].isin([NEWORDER, DELETE, CANCEL, EXECUTE_VISIBLE])]
	orders.reset_index(drop = True, inplace = True)

	while (orders.shape[0] > 0):
		idx = None
		if time:
			idx = getToProcess(orders, stride)
		else:
			## Advance forward 'stride' non-execute orders, or the end
			## of the dataset if none left
			each_status = np.cumsum(np.isin(np.array(orders)[:,1], [NEWORDER, DELETE, CANCEL]))
			idx = np.argmax(each_status > stride)
			idx = (idx) if idx != 0 else (orders.shape[0])
		
		to_process = np.array(orders.iloc[0:idx])
		orders.reset_index(drop = True, inplace = True)
		orders = orders.drop(range(0, min(idx, orders.shape[0])))
		processHistoricalOrders(to_process)
		print ("Remaining: %d" % (orders.shape[0]))

	CUM_VOLUME_EXP = np.cumsum(CUM_VOLUME_EXP)
	CUM_TRADES_EXP = np.cumsum(CUM_TRADES_EXP)

	d["CUM_VOLUME_EXP"] = CUM_VOLUME_EXP
	d["CUM_TRADES_EXP"] = CUM_TRADES_EXP
	d["PRICE_EXP"] = PRICE_EXP

'''
Method: makePlots
Arguments: d = dictionary of results, 
		   tag = string tag of saved plots
Returns: None

Description: 

'''
def makePlots(d, tag):
	BATCH_INTERVAL = d["BATCH_INTERVAL"]
	CUM_VOLUME_BATCH = d["CUM_VOLUME_BATCH"]
	CUM_TRADES_BATCH = d["CUM_TRADES_BATCH"]
	PRICE_BATCH = d["PRICE_BATCH"]
	BOOK_SIZE_BATCH = d["BOOK_SIZE_BATCH"]
	BATCHES = d["BATCHES"]

	CUM_VOLUME_EXP = d["CUM_VOLUME_EXP"]
	CUM_TRADES_EXP = d["CUM_TRADES_EXP"]
	PRICE_EXP = d["PRICE_EXP"]

	plt.plot(range(0, BATCHES), PRICE_BATCH, color = "blue", label = "Batched")
	plt.plot(range(0, len(PRICE_EXP)), PRICE_EXP, color = "red", label = "Reconstructed")
	plt.legend(loc = "lower left")
	plt.title("Market clearing price by # batches")
	plt.xlabel("Batch")
	plt.ylabel("Price")
	plt.savefig(str(tag) + "_Price_over_batch.png")

	plt.clf()
	plt.plot(range(0, BATCHES), CUM_VOLUME_BATCH, color = "blue", label = "Batched")
	plt.plot(range(0, len(CUM_VOLUME_EXP)), CUM_VOLUME_EXP, color = "red", label = "Reconstructed")
	plt.legend(loc = "upper left")
	plt.title("Cumulative volume traded by # batches")
	plt.xlabel("Batch")
	plt.ylabel("Volume")
	plt.savefig(str(tag) + "_Volume_over_batch.png")

	plt.clf()
	plt.plot(range(0, BATCHES), CUM_TRADES_BATCH, color = "blue", label = "Batched")
	plt.plot(range(0, len(CUM_TRADES_EXP)), CUM_TRADES_EXP, color = "red", label = "Reconstructed")
	plt.legend(loc = "upper left")
	plt.title("Cumulative trades by # batches")
	plt.xlabel("Batch")
	plt.ylabel("Trades")
	plt.savefig(str(tag) + "_Trades_over_batch.png")

	plt.clf()
	plt.plot(range(0, BATCHES), BOOK_SIZE_BATCH, color = "blue")
	plt.title("Periodic open order book size by # batches")
	plt.xlabel("Batch")
	plt.ylabel("Book size")
	plt.savefig(str(tag) + "_Book_over_batch.png")

	plt.clf()
	plt.plot(range(0, BATCHES), BATCH_INTERVAL, color = "blue")
	plt.title("Batch interval size (s) by batch #")
	plt.xlabel("Batch")
	plt.ylabel("Interval size (s)")
	plt.savefig(str(tag) + "_Interval_over_batch.png")

	output = np.array([range(0, BATCHES), CUM_VOLUME_BATCH, CUM_TRADES_BATCH, PRICE_BATCH, BOOK_SIZE_BATCH, BATCH_INTERVAL])
	output = pd.DataFrame(output.T, \
		columns = ['Batch No.', 'Cumulative Volume', 'Cumulative Trades', 'Price', 'Book Size', 'Interval (s)'])
	output.to_csv(str(tag) + ".csv", index = False)

'''
Method: loadOrderBook
Arguments: path = path to order book CSV file
Returns: None

Description: 

'''
def loadOrderBook(path):
	history = pd.read_csv(path, header = None, \
		names = ['Time', 'Event', 'OrderID', 'Size', 'Price', 'Direction'])
	return(history)

'''
Method: printHelp
Arguments: None
Returns: None

Description: Prints usage information for the batch reconstruction tool. 
'''
def printHelp():
	print ("\nUsage: python batchingSimulator.py [interval size] [--time_interval for time]\n")
	print ("First argument [interval size] is either the number of orders considered in a\n batch update, or the time interval of the update depending on second argument.\n")
	print ("Second argument is either --time_interval or --const_batch_size depending on desired\n simulation.\n")
	print ("Both arguments are necessary.\n")

###################################################################################################
###################################################################################################

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("stride", type=int)
	parser.add_argument("--time_interval", dest = 'time', action = 'store_true')
	parser.add_argument("--const_batch_size", dest = 'time', action = 'store_false')
	args = parser.parse_args()

	if args.stride == 0:
		printHelp()
		quit()

	path = "./MSFT_2012-06-21_message_10.csv"
	dat = loadOrderBook(path)

	manager = Manager()
	d = manager.dict()
	## Simulate batched results
	sim = Process(target = simFrequentBatchAuction, args = (d, dat, args.stride, args.time))
	sim.start()

	## Simulate reconstructed historical results
	rec = Process(target = reconstructOrderBook, args = (d, dat, args.stride, args.time))
	rec.start()
	sim.join(); rec.join()

	## Save output and plots
	tag = ("Time" + str(args.stride) + "s") if args.time else ("Stride" + str(args.stride))
	os.mkdir(tag)
	makePlots(d, "./" + tag + "/" + tag)

if __name__ == "__main__":
	main()