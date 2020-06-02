# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#https://github.com/rohitash-chandra
 

# ref: http://iamtrask.github.io/2015/07/12/basic-python-network/  
 

#Sigmoid units used in hidden and output  

# Numpy used: http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays
 
# this version will demonstrate momemntum and stocastic gradient descent 

#FNN for Time Series prediction and Regression
 

import matplotlib.pyplot as plt
import numpy as np 
import random
import time
 
class Network:

	def __init__(self, Topo, Train, Test, MaxTime,  MinPer, learnRate, use_stocasticGD, use_vanillalearning,  momentum_rate): 
		self.Top  = Topo  # NN topology [input, hidden, output]
		self.Max = MaxTime # max epocs
		self.TrainData = Train
		self.TestData = Test
		self.NumSamples = Train.shape[0]

		self.learn_rate  = learnRate
 

		self.minPerf = MinPer
		
		#initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
		np.random.seed() 
		self.W1 = np.random.uniform(-0.5, 0.5, (self.Top[0] , self.Top[1]))  
		#print(self.W1,  ' self.W1')
		self.B1 = np.random.uniform(-0.5,0.5, (1, self.Top[1])  ) # bias first layer
		#print(self.B1, ' self.B1')
		self.BestB1 = self.B1
		self.BestW1 = self.W1 
		self.W2 = np.random.uniform(-0.5, 0.5, (self.Top[1] , self.Top[2]))   
		self.B2 = np.random.uniform(-0.5,0.5, (1,self.Top[2]))  # bias second layer
		self.BestB2 = self.B2
		self.BestW2 = self.W2 
		self.hidout = np.zeros(self.Top[1] ) # output of first hidden layer
		self.out = np.zeros(self.Top[2]) #  output last layer

		self.hid_delta = np.zeros(self.Top[1] ) # output of first hidden layer
		self.out_delta = np.zeros(self.Top[2]) #  output last layer

		self.vanilla = use_vanillalearning  # canonical batch training mode - use full data set - no SGD  

		self.momenRate = momentum_rate

		self.stocasticGD = use_stocasticGD




	def sigmoid(self,x):
		return 1 / (1 + np.exp(-x))

	
	def softmax(self, x):
		# Numerically stable with large exponentials
		exps = np.exp(x - x.max())
		return exps / np.sum(exps, axis=0)

	def sampleEr(self,actualout):
		error = np.subtract(self.out, actualout)
		sqerror= np.sum(np.square(error))/self.Top[2] 
		 
		return sqerror

	def ForwardPass(self, X ): 
		z1 = X.dot(self.W1) - self.B1  
		self.hidout = self.sigmoid(z1) # output of first hidden layer   
		z2 = self.hidout.dot(self.W2)  - self.B2 
		self.out = self.sigmoid(z2)  # output second hidden layer



	def BackwardPass(self, input_vec, desired):   
		out_delta =   (desired - self.out)*(self.out*(1-self.out))  
		hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout)) 
		#https://www.tutorialspoint.com/numpy/numpy_dot.htm  https://www.geeksforgeeks.org/numpy-dot-python/
  
		if self.vanilla == True: #no momentum 
			self.W2+= self.hidout.T.dot(out_delta) * self.learn_rate
			self.B2+=  (-1 * self.learn_rate * out_delta)

			self.W1 += (input_vec.T.dot(hid_delta) * self.learn_rate) 
			self.B1+=  (-1 * self.learn_rate * hid_delta) 
		else: # use momentum
			v2 = self.W2.copy() #save previous weights http://cs231n.github.io/neural-networks-3/#sgd
			v1 = self.W1.copy()
			b2 = self.B2.copy()
			b1 = self.B1.copy()

			self.W2+= ( v2 *self.momenRate) + (self.hidout.T.dot(out_delta) * self.learn_rate)       # velocity update
			self.W1 += ( v1 *self.momenRate) + (input_vec.T.dot(hid_delta) * self.learn_rate)   
			self.B2+= ( b2 *self.momenRate) + (-1 * self.learn_rate * out_delta)       # velocity update
			self.B1 += ( b1 *self.momenRate) + (-1 * self.learn_rate * hid_delta)   
  
			
 
	def TestNetworkRegression(self, Data,  erTolerance):
		Input = np.zeros((1, self.Top[0])) # temp hold input
		Desired = np.zeros((1, self.Top[2])) 
		nOutput = np.zeros((1, self.Top[2]))
		testSize = Data.shape[0] 
		sse = 0  
		Input = np.zeros((1, self.Top[0])) # temp hold input

		predicted = np.zeros(testSize)

		self.W1 = self.BestW1
		self.W2 = self.BestW2 #load best knowledge
		self.B1 = self.BestB1
		self.B2 = self.BestB2 #load best knowledge
 
		for s in range(0, testSize):
							
			Input[:]  =   Data[s,0:self.Top[0]] 
			Desired[:] =  Data[s,self.Top[0]:] 

			self.ForwardPass(Input) 
			predicted[s] = self.out

			sse = sse+ self.sampleEr(Desired)   

		actual =  Data[:,self.Top[0]:]  



		return np.sqrt(sse/testSize),   actual, predicted




	def saveKnowledge(self):
		self.BestW1 = self.W1
		self.BestW2 = self.W2
		self.BestB1 = self.B1
		self.BestB2 = self.B2 

		#print (self.BestW1, self.BestW2, self.BestB1, self.BestB2)

	def BP_GD(self, trainTolerance):  


		Input = np.zeros((1, self.Top[0])) # temp hold input
		Desired = np.zeros((1, self.Top[2])) 

		minibatchsize = int(0.1* self.TrainData.shape[0]) # choose a mini-batch size for SGD

		er = [] # collect error over time
		epoch = 0
		bestRMSE= 10000 # assign a large number in begining to maintain best (lowest RMSE)
		bestTrain = 0
		while  epoch < self.Max and bestTrain < self.minPerf :
			sse = 0

			if(self.stocasticGD==True): # create a minibatch of samples 
				train_dat = np.array(self.TrainData).tolist()
				array = []
				for iteratable in range (0, minibatchsize):
					pat = random.randint(0, len(self.TrainData)-1) # construst a mini-batch for SGD
					array.append(train_dat[pat])		   	
				train_dat = np.asarray(array)
				num_batch = 10 # because your batch size is 10 %, you need 10 batches to cover full data approximately
			else:
				train_dat = self.TrainData
				num_batch = 1

			#print(train_dat)

			for batch in range(0, num_batch): # 10 mini batches in case of SGD. 1 major batch in case of GD

				for s in range(0, train_dat.shape[0]):
			
					Input[:]  =  train_dat[s,0:self.Top[0]]  
					Desired[:]  = train_dat[s,self.Top[0]:]  

					self.ForwardPass(Input)  
					self.BackwardPass(Input ,Desired)
					sse = sse+ self.sampleEr(Desired)
			 
			rmse = np.sqrt(sse/self.TrainData.shape[0]*self.Top[2])

			if rmse < bestRMSE:
				 bestRMSE = rmse
				 self.saveKnowledge() 
				 bestRMSE, actual, predicted = self.TestNetworkRegression(self.TrainData,   trainTolerance)
				 #print(bestRMSE )

			er = np.append(er, rmse)
			
			epoch=epoch+1  

		return (er, bestRMSE,  epoch, actual, predicted) 



def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]
	traindt = data[:,np.array(range(0,inputsize))]	
	dt = np.amax(traindt, axis=0)
	tds = abs(traindt/dt) 
	return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)

def main(): 
					
		
	problem = 3 # [1,2,3] choose your problem  
				
 


	if problem == 1:
		TrainData = np.loadtxt("data/Sunspot/train.txt") 
		TestData    = np.loadtxt("data/Sunspot/test.txt") 
		Hidden = 5
		Input = 4
		Output = 1 
		MaxTime = 1000   


	elif problem == 2:
		TrainData = np.loadtxt("data/Lazer/train.txt") 
		TestData    = np.loadtxt("data/Lazer/test.txt") 
		Hidden = 5
		Input = 4
		Output = 1 
		MaxTime = 1000   


	elif problem == 3:
		TrainData = np.loadtxt("data/Mackey/train.txt") 
		TestData    = np.loadtxt("data/Mackey/test.txt") 
		Hidden = 5
		Input = 4
		Output = 1 
		MaxTime = 1000  
	


	MinCriteria = 100 #stop when learn 100 percent - to ensure it does not stop ( does not apply for time series - regression problems)


	Topo = [Input, Hidden, Output] 
	MaxRun = 5 # number of experimental runs 
	 

	trainTolerance = 0.2 # [eg 0.15 would be seen as 0] [ 0.81 would be seen as 1]
	testTolerance = 0.49

	learnRate = 0.1  

	useStocastic = False # False for vanilla BP. True for Stocastic BP
	updateStyle = True # True for Vanilla (Canonical) Gradient Descent, False for Gradient Descent with momentum  

	momentum_rate = 0.001 # 0.1 ends up having very large weights. you can try and see
	 
 
 

	trainRMSE =  np.zeros(MaxRun)
	testRMSE =  np.zeros(MaxRun)
	Epochs =  np.zeros(MaxRun)
	Time =  np.zeros(MaxRun)

	for run in range(0, MaxRun  ):
		print(run, ' is experimental run') 

		fnn = Network(Topo, TrainData, TestData, MaxTime,   MinCriteria, learnRate, useStocastic, updateStyle, momentum_rate)
		start_time=time.time()
		(erEp,  trainRMSE[run],  Epochs[run], actual, predicted) = fnn.BP_GD(trainTolerance)   


		Time[run]  =time.time()-start_time
		(testRMSE[run], actual, predicted) = fnn.TestNetworkRegression(TestData,  testTolerance) 


 
	print('RMSE performance for each experimental run') 
	print(trainRMSE)
	print(testRMSE)
	print('Epocs and Time taken for each experimental run') 
	print(Epochs)
	print(Time)
	print('mean and std of classification performance') 
	
	print(np.mean(trainRMSE), np.std(trainRMSE))
	print(np.mean(testRMSE), np.std(testRMSE))

	print(' print mean and std of computational time taken') 
	
	print(np.mean(Time), np.std(Time))
	
	# fig of last run
				 
	plt.figure()
	plt.plot(erEp )
	plt.ylabel('error')  
	plt.savefig('out.png')
	plt.clf()

	y_train = np.linspace(0, actual.shape[0], num=actual.shape[0])


	plt.figure()
	plt.plot(y_train, actual, label='actual')
	plt.plot(y_train, predicted, label='predicted')
	plt.ylabel('RMSE')  
	plt.xlabel('Time (samples)')  
	plt.savefig('pred_timeseries'+str(problem)+'.png')
			 
 
if __name__ == "__main__": main()

