
import numpy as np
import struct, math, random, sys
#import copy

#multi processing
from multiprocessing import Pool, Manager, Process
#from multiprocessing.managers import BaseManager

from collections import deque

NUM_PIXELS = 784
PIC_ROWS = 1 #28
PIC_COLS = 784 #28
OUT_ROWS = 1
OUT_COLS = 10

LEARNING_RATE = 5 #10**5??
NUM_TRAIN_STEPS = 100 #2000 for grad descent non-empirical, 50 for annealing?

NUM_ANNEALING_STEPS = 50
SCHEDULE = [1/(i+1) for i in xrange(NUM_ANNEALING_STEPS)]
PROPOSAL_SCALE = .5

 #10000
NUM_TESTS = 10000

MESH_ROWS = 3
MESH_COLS = 4



# POOL_THREAD_COUNT = 5

MULTI_PROCESS = False # I think there are still issues with multiprocessing...
LINEAR_ERROR = False #This Cant be used with non-empirical Gradient Descent!!!!!!!!!!
GRAD_DESCENT = True
GRAD_DESCENT_EMPIRICAL = False #wehter to rerun the forward pass to calculate the derivative here
MAX_RECUR = 1000

RESET_NO_ENERGY = False # whether to reset to trained init values, or just to 0 (no energy)

#MAX_E_POW = 10

MAX_ENERGY_OUT = 1 #the maximum fraction of a cell's enegery it can transfer in total
REFRACTORY_PERIOD = 1 #number of timesteps until cell can fire again
EPS = .01
AE = .5 #activation energy
INIT_AE_DIST = .2 #cells can start at a max of AE-INIT_AE_DIST
RANDOM_START_INIT_E = True #whether to start init val at 0 or at START_INIT_VAL
START_INIT_VAL = 0 #energey starts empty if 
RUN_TIME = 1 #on each forward pass, will be run  RUN_TIME*(width+height) times


# needs to be global for pools, but swtiched to single process
# def globalGetVarDeltaNoReturn(mesh, dataArray2D, row, col, error, inputArray2D, errorFunc, wrapper):
# 	wrapper[(row, col)] = mesh.getVarDelta(dataArray2D, row, col, error, inputArray2D, errorFunc)

# Test becuase shared memory not working
# def test(mesh1, mesh2, otherarg, testList):
# 	print "lol"
# 	testList[0] = 1


class Neuron:
	def __init__(self, activationEnergy, inputRows, inputCols, outputRows, outputCols, selfRow, selfCol, refractoryPeriod):
		self.derivative = 0 #derivative of error with respect to the val stored in the neuron
		self.timeToFiredIn = {} #dict of timestep to list of neurons that fired in to it (children for back prop)

		self.selfRow = selfRow
		self.selfCol = selfCol

		self.inputCols = inputCols
		self.inputRows = inputRows
		self.outputRows = outputRows
		self.outputCols = outputCols

		self.refractoryPeriod = refractoryPeriod
		self.currentPeriodLeft = 0
		self.e = activationEnergy
		self.val = 0
		self.valDelta = 0 #val is not a variable that is learned. it is only updated in each timestep simulated.

		#bias terms
		self.initialVal = [[START_INIT_VAL if not RANDOM_START_INIT_E else random.random()]] #List of list generalizes easier.
		self.initialValDelta = [[0]]

		#try starting these randomly
		self.dirToWeightOut = [ map(lambda _: random.random(),[0, 0, 0, 0, 0, 0, 0, 0]) ] #index 0 is up. 1 is top right. And so on. 
		self.inputCoordToWeightIn = [ map(lambda _: random.random()/(inputRows*inputCols),[0 for _ in xrange(inputCols)]) for _ in xrange(inputRows) ] #weights from input. scale down so neuron val starts around .5
		self.outputCoordToWeightOut = [ map(lambda _: random.random(),[0 for _ in xrange(outputCols)]) for _ in xrange(outputRows) ] #weights to output
		
		#changes to be made to above vars
		self.dirToWeightOutDelta = [[0, 0, 0, 0, 0, 0, 0, 0]]
		self.inputCoordToWeightInDelta = [[0 for _ in xrange(inputCols)] for _ in xrange(inputRows)]
		self.outputCoordToWeightOutDelta = [[0 for _ in xrange(outputCols)] for _ in xrange(outputRows)]
	
	# def __reduce__(self):
	# 	return (neuralMesh, (self.selfRow, self.selfCol, self.e, self.val, self.valDelta, self.initialVal, self.initialValDelta, self.dirToWeightOut, self.inputCoordToWeightIn, self.outputCoordToWeightOut, self.dirToWeightOutDelta, self.inputCoordToWeightInDelta, self.outputCoordToWeightOutDelta))

	def resetEnergy(self, useDeltas):
		if not RESET_NO_ENERGY:
			self.val = self.initialVal[0][0] if not useDeltas else min(max(0,self.initialVal[0][0]+ self.initialValDelta[0][0]), AE-INIT_AE_DIST)
		else:
			self.val = 0

	def updateValueByDelta(self):
		self.val += self.valDelta
		self.valDelta = 0

	def setDeltasRand(self):
		self.dirToWeightOutDelta = [ map(lambda _: random.uniform(-PROPOSAL_SCALE,PROPOSAL_SCALE),[0, 0, 0, 0, 0, 0, 0, 0]) ] 
		self.inputCoordToWeightInDelta = [ map(lambda _: random.uniform(-PROPOSAL_SCALE,PROPOSAL_SCALE)/(self.inputRows*self.inputCols),[0 for _ in xrange(self.inputCols)]) for _ in xrange(self.inputRows) ]
		self.outputCoordToWeightOutDelta = [ map(lambda _: random.uniform(-PROPOSAL_SCALE,PROPOSAL_SCALE),[0 for _ in xrange(self.outputCols)]) for _ in xrange(self.outputRows) ] 
		self.initialValDelta = [[random.uniform(-PROPOSAL_SCALE,PROPOSAL_SCALE)]]

	#makes the suggested learning changes, then resets delta counter
	def updateVarByDeltas(self):
		#update initial value
		# print "self.initialValDelta: ", self.initialValDelta[0][0]
		# print "self.dirToWeightOutDelta: ", self.dirToWeightOutDelta[0][0]
		self.initialVal[0][0] += self.initialValDelta[0][0]
		self.initialValDelta[0][0] = 0
		self.initialVal[0][0] = max(0, self.initialVal[0][0]) #shouldnt go below 0
		self.initialVal[0][0] = min(AE-INIT_AE_DIST, self.initialVal[0][0]) #should go above this. prevents init then fire immediately with no input

		#update mesh weights
		for i in xrange(len(self.dirToWeightOutDelta[0])):
			self.dirToWeightOut[0][i] += self.dirToWeightOutDelta[0][i] #make appropriate change
			self.dirToWeightOutDelta[0][i] = 0 #reset delta
			self.dirToWeightOut[0][i] = max(0, self.dirToWeightOut[0][i])

		#update input weights
		for r in xrange(len(self.inputCoordToWeightInDelta)):
			for c in xrange(len(self.inputCoordToWeightInDelta[r])):
				self.inputCoordToWeightIn[r][c] += self.inputCoordToWeightInDelta[r][c]
				self.inputCoordToWeightInDelta[r][c] = 0
				self.inputCoordToWeightIn[r][c] = max(0, self.inputCoordToWeightIn[r][c])

		#update output weights
		for r in xrange(len(self.outputCoordToWeightOutDelta)):
			for c in xrange(len(self.outputCoordToWeightOutDelta[r])):
				self.outputCoordToWeightOut[r][c] += self.outputCoordToWeightOutDelta[r][c]
				self.outputCoordToWeightOutDelta[r][c] = 0
				self.outputCoordToWeightOut[r][c] = max(0, self.outputCoordToWeightOut[r][c])

	def putIn(self, inputArray2D, useDeltas):
		for r in xrange(len(inputArray2D)):
			for c in xrange(len(inputArray2D[r])):

				inWeight = self.inputCoordToWeightIn[r][c]
				if useDeltas:
					inWeight += self.inputCoordToWeightInDelta[r][c]
					inWeight = max(0, inWeight)

				self.val += inWeight * inputArray2D[r][c]

	def putOut(self, outputArray2D, useDeltas):
		for r in xrange(len(outputArray2D)):
			for c in xrange(len(outputArray2D[r])):

				outWeight = self.outputCoordToWeightOut[r][c]
				if useDeltas:
					outWeight += self.outputCoordToWeightOutDelta[r][c]
					outWeight = max(0, outWeight)

				outputArray2D[r][c] += self.val * outWeight

	#fires a neron to change adjacent deltas
	def run(self, neuronArray, useDeltas, timeStep):
		#still has time left in refractory period
		if self.currentPeriodLeft > 0:
			self.currentPeriodLeft -= 1
			return

		#not enough energy yet
		if self.val < self.e:
			return

		#FIRE
		numR = len(neuronArray) #number of rows in mesh
		numC = len(neuronArray[0]) #number of cols in mesh
		#iterate over adjacent neurons
		for deltaR in xrange(-1,2):
			for deltaC in xrange(-1,2):
				if deltaR == 0 and deltaC == 0: #looking at self position
					continue
				r = self.selfRow + deltaR 
				c = self.selfCol + deltaC
				# print "deltaR, deltaC:", deltaR, deltaC
				# if not out of bound, fire that way
				if Neuron.rowColInRange(r, c, numR, numC):
					# print "r:", r
					# print "c:", c
					# print "numR:", numR
					# print "numC:", numC
					# print "neuronArray[r][c].valDelta," , neuronArray[r][c].valDelta
					# print "Neuron.directionIndex(deltaR, deltaC):", Neuron.directionIndex(deltaR, deltaC)
					# print "len(self.dirToWeightOut):", len(self.dirToWeightOut)
					# Weights based on whether using deltas:
					weightsOut = [(max(0,self.dirToWeightOut[0][i] + self.dirToWeightOutDelta[0][i]) if useDeltas else self.dirToWeightOut[0][i]) for i in xrange(len(self.dirToWeightOut[0]))]
					neuronArray[r][c].valDelta += self.val * Neuron.scaleDownWeights(weightsOut)[Neuron.directionIndex(deltaR, deltaC)]
					if not timeStep in neuronArray[r][c].timeToFiredIn:
						neuronArray[r][c].timeToFiredIn[timeStep] = []
					neuronArray[r][c].timeToFiredIn[timeStep].append(self)

		#reset val of this fired neuron
		self.resetEnergy(useDeltas)
		#restart refreactory period
		self.currentPeriodLeft = self.refractoryPeriod

	def printNeuron(self):
		print "\nNeuron at Row ", self.selfRow, " and Col ", self.selfCol, ":"
		print "activation energy: ", self.e
		print "current energy stored: ", self.val
		print "initial energy at reset: ", self.initialVal[0][0]
		print "sum of wieghts to other neurons: ", sum(self.dirToWeightOut[0])
		print "sum of wieghts from input: ", sum(map(sum, self.inputCoordToWeightIn))
		print "sum of wieghts from output: ", sum(map(sum, self.outputCoordToWeightOut))

	#if we are firing more energy than we have, this is a problem (prevent spontaneous eneergy production)
	@staticmethod
	def scaleDownWeights(weights):
		totalEnergyFraction = sum(weights)
		if totalEnergyFraction <= MAX_ENERGY_OUT:
			return weights
		return map(lambda num: MAX_ENERGY_OUT*num/totalEnergyFraction, weights) #normalize then mult my fraction

	@staticmethod
	def rowColInRange(r, c, numR, numC):
		return r >= 0 and c >= 0 and r < numR and c < numC

	@staticmethod
	#figure out direction index based on row delta and col delta
	def directionIndex(deltaR, deltaC):
		if deltaR == -1 and deltaC == 0:
			return 0
		elif deltaR == -1 and deltaC == 1:
			return 1
		elif deltaR == 0 and deltaC == 1:
			return 2
		elif deltaR == 1 and deltaC == 1:
			return 3
		elif deltaR == 1 and deltaC == 0:
			return 4
		elif deltaR == 1 and deltaC == -1:
			return 5
		elif deltaR == 0 and deltaC == -1:
			return 6
		elif deltaR == -1 and deltaC == -1:
			return 7






#Neural net: input -> grid of neurons connected to adjacent neurons -> output. 2D input, mesh, and output implementation.
class neuralMesh:
	def __init__(self, rows, cols, inputRows, inputCols, outputRows, outputCols, activationEnergy, eps, learningRate, runtime, refractoryPeriod):
		self.eps = eps #epsilon
		self.runtime = runtime
		self.learningRate = learningRate
		self.rows = rows
		self.cols = cols
		self.inputRows = inputRows
		self.inputCols = inputCols
		self.outputRows = outputRows
		self.outputCols = outputCols

		#self.multiManager = Manager()
		#add "bm." to neuron ro make neurons shared memory for multi
		# BaseManager.register('neuron', neuron)
		# bm = BaseManager()
		# bm.start()
		#instatiate bm.neuron ...
		self.neuronArray = [[Neuron(activationEnergy, inputRows, inputCols, outputRows, outputCols, r, c, refractoryPeriod) for c in xrange(cols)] for r in xrange(rows)] 

	def trainStepGradDescentMNIST(self, inputArray2D, correctLabel):
		probabilities = self.forwardPass(inputArray2D, False)[0] ## Run the forward pass

		maxProb = max(probabilities)
		if maxProb == 0:
			maxProb = 1;
		probabilitiesNormalized = map(lambda x: x/maxProb, probabilities)

		normalizedDerives = [probabilitiesNormalized[i] for i in xrange(self.outputCols)]
		normalizedDerives[correctLabel] -= 1

		logitOutDerivatives = [normalizedDerives[i]/maxProb for i in xrange(self.outputCols)] 

		#update output weight deltas
		for row in self.neuronArray:
			for neuron in row:
				for i in xrange(self.outputCols):
					neuron.outputCoordToWeightOutDelta[0][i] = neuron.val*logitOutDerivatives[i] * -self.learningRate
					neuron.derivative += logitOutDerivatives[i]*neuron.outputCoordToWeightOut[0][i]



		#### do a breathfirst search on neurons to do back prop recurrently ####

		neuronQue = deque([])
		for row in self.neuronArray:
			for neuron in row:
				neuronQue.appendleft(neuron)
		currentStep = self.runtime*(self.rows+self.cols)-1
		recursionDepth = 0
		while neuronQue:
			if recursionDepth >= MAX_RECUR:
				break
			recursionDepth += 1

			neuron = neuronQue.pop()

			neuron.initialValDelta[0][0] = neuron.derivative * -self.learningRate
			
			#### for each direction (weight coming in), calculate change in the weigth coming out of it into your neuron. 
			#     also update the derivative there
			children = neuron.timeToFiredIn[currentStep] if currentStep in neuron.timeToFiredIn else []
			for child in children:
				otherNeuron = child
				deltaR = otherNeuron.selfRow-neuron.selfRow
				deltaC = otherNeuron.selfCol-neuron.selfCol
				# print "deltaR, deltaC:", deltaR, deltaC
				direction = Neuron.directionIndex(deltaR,deltaC)
				weightOut = otherNeuron.dirToWeightOut[0][direction]
				sumOfOtherWeights = sum(otherNeuron.dirToWeightOut[0])
				dNormalizedWeightdWeight = sumOfOtherWeights/((sumOfOtherWeights+weightOut)**2)
				## set derivative for this weight
				otherNeuron.dirToWeightOutDelta[0][direction] += neuron.derivative*otherNeuron.val*dNormalizedWeightdWeight * -self.learningRate
				otherNeuron.derivative += neuron.derivative*(weightOut/sumOfOtherWeights)
			###

			#add Children
			for child in children:
				neuronQue.appendleft(child) 

			currentStep -= 1

		####			######



		## update input weight deltas
		for row in self.neuronArray:
			for neuron in row:
				for i in xrange(self.inputCols):
					neuron.inputCoordToWeightInDelta[0][i] += neuron.derivative*inputArray2D[0][i] * -self.learningRate




		#### MAKE ALL those changes and reset deltas
		for row in self.neuronArray:
			for neuron in row:
				neuron.updateVarByDeltas()


		return [probabilities]

	# does grad descent empircally. returns error before the trainstep.
	def trainStepGradDescentEmpirical(self, inputArray2D, errorFunc):
		error = errorFunc(self.forwardPass(inputArray2D, False))

		#calculate changes to make to each variable by testing change of eps
		for row in self.neuronArray:
			for neuron in row:
				if MULTI_PROCESS:
					#run all the testing forward passes (bulk of them) in seperate threads now:
					self.setNeuronDeltasMultiProcessed(neuron, error, inputArray2D, errorFunc)
				else:
					self.setNeuronDeltas(neuron, error, inputArray2D, errorFunc)
				
		#make those changes and reset deltas
		for row in self.neuronArray:
			for neuron in row:
				#single thread: 
				neuron.updateVarByDeltas()

		return error

	# temp is how much worse the error can be. simulated annealing.
	def trainStepRandNoise(self, inputArray2D, errorFunc):
		originalError = errorFunc(self.forwardPass(inputArray2D, False))
		oldError = originalError
		for step in xrange(NUM_ANNEALING_STEPS): 
			temp = SCHEDULE[step]
			#random proposal func
			for row in self.neuronArray:
				for neuron in row:
					neuron.setDeltasRand()
			#run forward pass adding on deltas
			newError = errorFunc(self.forwardPass(inputArray2D, True))
			#accept proposal
			if newError <= oldError+temp:
				# print "new error: ", newError
				oldError = newError
				for row in self.neuronArray:
					for neuron in row:
						neuron.updateVarByDeltas()
		finalError = oldError
		return (originalError, finalError)
				

	#multi processes
	# def multiprocessFunc(func, argTuple):
	# 	pool = Pool(POOL_THREAD_COUNT)
	# 	pool.apply_async(func, argTuple)

	#shared list of lists NOT WORKING use dict instead
	# #test each of the vars
	# #multi processed
	# def setNeuronDeltasMultiProcessed(self, neuron, error, inputArray2D, errorFunc):
	# 	# a place we can stick these deltas asycnhronously #
	# 	initDeltaWrapper = self.multiManager.list([[0]])
	# 	dirDeltaWrapper = self.multiManager.list([[0, 0, 0, 0, 0, 0, 0, 0]])
	# 	inDeltaWrapper = self.multiManager.list([[0 for _ in xrange(self.inputCols)] for _ in xrange(self.inputRows)])
	# 	outDeltaWrapper = self.multiManager.list([[0 for _ in xrange(self.outputCols)] for _ in xrange(self.outputRows)])

	# 	## stick delas in right spot ##
	# 	pool = Pool(POOL_THREAD_COUNT)
	# 	#set init value delta
	# 	pool.apply_async(self.getVarDeltaNoReturn, (neuron.initialVal, 0, 0, error, inputArray2D, errorFunc, initDeltaWrapper))

	# 	#set mesh weight deltas
	# 	for c in xrange(len(neuron.dirToWeightOut[0])):
	# 		pool.apply_async(self.getVarDeltaNoReturn, (neuron.dirToWeightOut, 0, c, error, inputArray2D, errorFunc, dirDeltaWrapper))

	# 	#set input weights deltas
	# 	for r in xrange(len(neuron.inputCoordToWeightIn)):
	# 		for c in xrange(len(neuron.inputCoordToWeightIn[r])):
	# 			pool.apply_async(self.getVarDeltaNoReturn, (neuron.inputCoordToWeightIn, r, c, error, inputArray2D, errorFunc, inDeltaWrapper))

	# 	#set output weights deltas
	# 	for r in xrange(len(neuron.outputCoordToWeightOut)):
	# 		for c in xrange(len(neuron.outputCoordToWeightOut[r])):
	# 			pool.apply_async(self.getVarDeltaNoReturn, (neuron.outputCoordToWeightOut, r, c, error, inputArray2D, errorFunc, outDeltaWrapper))

	# 	pool.close()
	# 	pool.join()

	# 	## set all deltas ##
	# 	for r in xrange(len(initDeltaWrapper)):
	# 		for c in xrange(len(initDeltaWrapper[r])):
	# 			neuron.initialValDelta[r][c] = initDeltaWrapper[r][c]

	# 	for r in xrange(len(dirDeltaWrapper)):
	# 		for c in xrange(len(dirDeltaWrapper[r])):
	# 			neuron.dirToWeightOutDelta[r][c] = dirDeltaWrapper[r][c]

	# 	for r in xrange(len(inDeltaWrapper)):
	# 		for c in xrange(len(inDeltaWrapper[r])):
	# 			neuron.inputCoordToWeightInDelta[r][c] = inDeltaWrapper[r][c]

	# 	for r in xrange(len(outDeltaWrapper)):
	# 		for c in xrange(len(outDeltaWrapper[r])):
	# 			neuron.outputCoordToWeightOutDelta[r][c] = outDeltaWrapper[r][c]


	# def getVarDeltaNoReturn(self, dataArray2D, row, col, error, inputArray2D, errorFunc, answerArray):
	# 	answerArray[row][col] = getVarDelta(dataArray2D, row, col, error, inputArray2D, errorFunc)

	# def __reduce__(self):
	# 	return (neuralMesh, (self.eps, self.runtime, self.learningRate, self.rows, self.cols, self.inputRows, self.inputCols, self.outputRows, self.outputCols, self.neuronArray))

	#test each of the vars
	#multi processed
	def setNeuronDeltasMultiProcessed(self, neuron, error, inputArray2D, errorFunc):
		# a place we can stick these deltas asycnhronously #
		multiManager = Manager()
		initDeltaWrapper = multiManager.dict() #tuple of coord to correct delta value. functions like a 2D array.
		dirDeltaWrapper = multiManager.dict()
		inDeltaWrapper = multiManager.dict()
		outDeltaWrapper = multiManager.dict()

		## stick delas in right spot ##
		# pool = Pool(POOL_THREAD_COUNT)

		# #Test becuase shared memory not working
		# testList = multiManager.list([0])
		# pool.apply_async(test, (5, testList))
		# # pool.apply_async(test, (copy.deepcopy(self), testList)) 
		# pool.apply_async(test, (self, neuralMesh(1, 1, 1, 1, 1, 1, 1, 1, 1, 1), [[]], testList))
		# pool.close()
		# pool.join()
		# # n = neuralMesh(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
		# # p = Process(target= test, args= (self, n, testList))
		# # p.start()
		# # p.join()
		# print testList

		processList = []

		#set init value delta
		# not sure why, but seems we need to use process, not pool... so
		# pool.apply_async(globalGetVarDeltaNoReturn, (self, neuron.initialVal, 0, 0, error, inputArray2D, errorFunc, initDeltaWrapper))
		processList.append(Process(target= self.getVarDeltaNoReturn, args= (neuron.initialVal, 0, 0, error, inputArray2D, errorFunc, initDeltaWrapper)))

		#set mesh weight deltas
		for c in xrange(len(neuron.dirToWeightOut[0])):
			# pool.apply_async(globalGetVarDeltaNoReturn, (self, neuron.dirToWeightOut, 0, c, error, inputArray2D, errorFunc, dirDeltaWrapper))
			processList.append(Process(target= self.getVarDeltaNoReturn, args= (neuron.dirToWeightOut, 0, c, error, inputArray2D, errorFunc, dirDeltaWrapper)))

		#set input weights deltas
		for r in xrange(len(neuron.inputCoordToWeightIn)):
			for c in xrange(len(neuron.inputCoordToWeightIn[r])):
				# pool.apply_async(globalGetVarDeltaNoReturn, (self, neuron.inputCoordToWeightIn, r, c, error, inputArray2D, errorFunc, inDeltaWrapper))
				processList.append(Process(target= self.getVarDeltaNoReturn, args= (neuron.inputCoordToWeightIn, r, c, error, inputArray2D, errorFunc, inDeltaWrapper)))

		#set output weights deltas
		for r in xrange(len(neuron.outputCoordToWeightOut)):
			for c in xrange(len(neuron.outputCoordToWeightOut[r])):
				# pool.apply_async(globalGetVarDeltaNoReturn, (self, neuron.outputCoordToWeightOut, r, c, error, inputArray2D, errorFunc, outDeltaWrapper))
				processList.append(Process(target= self.getVarDeltaNoReturn, args= (neuron.outputCoordToWeightOut, r, c, error, inputArray2D, errorFunc, outDeltaWrapper)))

		# pool.close()
		# pool.join()
		map(lambda proc: proc.start(), processList)
		map(lambda proc: proc.join(), processList)
		# delete all spawned processes to prevent to many
		for proc in processList:
			del proc
		del processList

		## set all deltas ##
		# print initDeltaWrapper[(0,0)]
		neuron.initialValDelta[0][0] = initDeltaWrapper[(0,0)]

		for c in xrange(len(neuron.dirToWeightOutDelta[0])):
			neuron.dirToWeightOutDelta[0][c] = dirDeltaWrapper[(0,c)]

		for r in xrange(len(neuron.inputCoordToWeightInDelta)):
			for c in xrange(len(neuron.inputCoordToWeightInDelta[r])):
				neuron.inputCoordToWeightInDelta[r][c] = inDeltaWrapper[(r,c)]

		for r in xrange(len(neuron.outputCoordToWeightOutDelta)):
			for c in xrange(len(neuron.outputCoordToWeightOutDelta[r])):
				neuron.outputCoordToWeightOutDelta[r][c] = outDeltaWrapper[(r,c)]

	def getVarDeltaNoReturn(self, dataArray2D, row, col, error, inputArray2D, errorFunc, wrapper):
		wrapper[(row, col)] = self.getVarDelta(dataArray2D, row, col, error, inputArray2D, errorFunc)

	#test each of the vars
	def setNeuronDeltas(self, neuron, error, inputArray2D, errorFunc):
		#set init value delta
		neuron.initialValDelta[0][0] = self.getVarDelta(neuron.initialVal, 0, 0, error, inputArray2D, errorFunc)

		#set mesh weight deltas
		for c in xrange(len(neuron.dirToWeightOut[0])):
			neuron.dirToWeightOutDelta[0][c] = self.getVarDelta(neuron.dirToWeightOut, 0, c, error, inputArray2D, errorFunc)

		#set input weights deltas
		for r in xrange(len(neuron.inputCoordToWeightIn)):
			for c in xrange(len(neuron.inputCoordToWeightIn[r])):
				neuron.inputCoordToWeightInDelta[r][c] = self.getVarDelta(neuron.inputCoordToWeightIn, r, c, error, inputArray2D, errorFunc)

		#set output weights deltas
		for r in xrange(len(neuron.outputCoordToWeightOut)):
			for c in xrange(len(neuron.outputCoordToWeightOut[r])):
				neuron.outputCoordToWeightOutDelta[r][c] = self.getVarDelta(neuron.outputCoordToWeightOut, r, c, error, inputArray2D, errorFunc)

	def getVarDelta(self, dataArray2D, row, col, error, inputArray2D, errorFunc):
		# print "Old Var val:", dataArray2D[row][col]
		dataArray2D[row][col] += self.eps #change var a bit
		# print "New Var val:", dataArray2D[row][col]
		newError = errorFunc(self.forwardPass(inputArray2D, False)) #see the difference
		dataArray2D[row][col] -= self.eps #change back
		slope = (newError - error)/self.eps
		# print "(newError - error):", (newError - error)
		# if newError != error:
		# 	print "old error", error
		# 	print "new error", newError
		# 	print "chane by: ", -slope*self.learningRate
		return -slope*self.learningRate #decrease delta based on slope (head away)


	def forwardPass(self, inputArray2D, useDeltas):
		#reset energies
		for row in self.neuronArray:
			for neuron in row:
				neuron.resetEnergy(useDeltas)

		#init energies
		for row in self.neuronArray:
			for neuron in row:
				neuron.putIn(inputArray2D, useDeltas)

		#update frames for each timestep by firing then updating each neuron repeatedly
		for timestep in xrange(self.runtime*(self.rows+self.cols)):
			#fire all
			for row in self.neuronArray:
				for neuron in row:
					neuron.run(self.neuronArray, useDeltas, timestep)
			#update all as indicated by fires
			for row in self.neuronArray:
				for neuron in row:
					neuron.updateValueByDelta()

		#construct output
		outputArray2D = [[0 for _ in xrange(self.outputCols)] for _ in xrange(self.outputRows)]
		for row in self.neuronArray:
			for neuron in row:
				neuron.putOut(outputArray2D, useDeltas)

		return outputArray2D

	def printAllNeuron(self):
		for row in self.neuronArray:
			for neuron in row:
				neuron.printNeuron()









####TODO: 
### GET SOME RESULTS then CALCULATE DERIVATIVE INSTEAD
### fix multiprocessing?
### linear error not great. non-linear e power is too large?


def crossEntropyMNIST(outputArray2D, label):
	initialOutput = outputArray2D[0]
	return -math.log(softMaxMNIST(initialOutput)[label])

def softMaxMNIST(initialOutput1D):
	# print "initialOutput[i]:", initialOutput[0]
	# eRaisedToPowers = [math.e**min(MAX_E_POW, initialOutput1D[i]) for i in xrange(len(initialOutput1D))]
	
	# Normalize by dividin by the max because e pows are too large
	maximum = max(initialOutput1D)
	if maximum == 0:
		maximum = 1
	eRaisedToPowers = [math.e**(initialOutput1D[i]/maximum) for i in xrange(len(initialOutput1D))]
	sumEs = sum(eRaisedToPowers)
	normalized = map(lambda x: x/sumEs, eRaisedToPowers)
	return normalized


def normalizeLinear(initialOutput1D):
	minimum = min(initialOutput1D)
	shiftedOutput = map(lambda x: x-minimum, initialOutput1D) #add -min to everything
	total = sum(shiftedOutput)
	normalized = map(lambda x: x/total, shiftedOutput)
	return normalized

def linearError(outputArray2D, label):
	initialOutput = outputArray2D[0]
	return 1-normalizeLinear(initialOutput)[label]



def main():


	nm = neuralMesh(MESH_ROWS, MESH_COLS, PIC_ROWS, PIC_COLS, 
		OUT_ROWS, OUT_COLS, AE, EPS, LEARNING_RATE, RUN_TIME, REFRACTORY_PERIOD)



	## train ##

	imageTrainFile = open("train-images.idx3-ubyte", "rb")
	labelTrainFile = open("train-labels.idx1-ubyte", "rb")
	imageTrainFile.read(16) #get rid of header
	labelTrainFile.read(8)

	sumErrorChange = 0
	for i in xrange(NUM_TRAIN_STEPS):
		#read in data

		#image data
		byteData = imageTrainFile.read(NUM_PIXELS)
		imgBytes = struct.unpack("784B", byteData)
		imgInput = [(np.array(imgBytes, dtype=float)/255).tolist()]

		#label data
		bytesForLabel = labelTrainFile.read(1)
		label = int(struct.unpack("1B", bytesForLabel)[0])


		#train
		if LINEAR_ERROR:
			errorFunc = (lambda x: linearError(x, label))
		else:
			errorFunc = (lambda x: crossEntropyMNIST(x, label))

		if GRAD_DESCENT_EMPIRICAL:
			curError = nm.trainStepGradDescentEmpirical(imgInput, errorFunc) #TRAIN STEP
			print "current error: ", curError
			print "before step: ", i
			# print "current mesh: "
			# nm.printAllNeuron()
			nm.neuronArray[0][0].printNeuron()
			print ""
		elif GRAD_DESCENT:
			probabilitiesBefore = nm.trainStepGradDescentMNIST(imgInput, label) #TRAIN STEP
			errorBefore = crossEntropyMNIST(probabilitiesBefore, label)
			currentError = crossEntropyMNIST(nm.forwardPass(imgInput, False), label)
			print "current error: ", currentError
			print "at step: ", i
			print "change of:", currentError-errorBefore
			sumErrorChange += currentError-errorBefore
			print "total of error deltas after each step:", sumErrorChange
			nm.neuronArray[0][0].printNeuron()
			print ""
		else:
			errors = nm.trainStepRandNoise(imgInput, errorFunc) #TRAIN STEP
			print "anneal step ", i
			print "startError: ", errors[0]
			print "finalError: ", errors[1]
			print "change: ", errors[1]-errors[0]
			nm.neuronArray[0][0].printNeuron()
			print ""

		


	## test ##

	imageTestFile = open("t10k-images.idx3-ubyte", "rb")
	labelTestFile = open("t10k-labels.idx1-ubyte", "rb")
	bytesForImage = imageTestFile.read(16) #get rid of header
	bytesForImage = labelTestFile.read(8)
	numCorrect = 0
	for i in xrange(NUM_TESTS):
		# read in data

		#image data
		byteData = imageTrainFile.read(NUM_PIXELS)
		imgBytes = struct.unpack("784B", byteData)
		imgInput = [(np.array(imgBytes, dtype=float)/255).tolist()]

		#label
		bytesForLabel = labelTrainFile.read(1)
		label = int(struct.unpack("1B", bytesForLabel)[0])


		# test

		netOutput = nm.forwardPass(imgInput, False) #TEST STEP
		if LINEAR_ERROR:
			probabilities = normalizeLinear(netOutput[0])
		else:
			probabilities = softMaxMNIST(netOutput[0])
		prediction = max(range(10), key=lambda i: probabilities[i])
		#old code: np.argmax(probabilities)

		if label == prediction:
			numCorrect += 1

		print "test number:", i
		print "accuracy avg so far:", float(numCorrect)/float(i+1)


if __name__ == "__main__":
	main()