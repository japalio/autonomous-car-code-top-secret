import random
import math
import hashlib

def isHeavyHitter(minFrequency):
	if minFrequency > 879:
		return True
	else:
		return False

def createDataStream():
	dataStream = []
	for i in range(9):
		freq = i + 1
		for j in range(1000):
			value = j + i * 1000
			dataStream.append((value, freq))
	for j in range(50):
		i = j + 1
		freq = math.pow(i,2)
		value = 9000 + i
		dataStream.append((value, freq))

	return dataStream

def nonDecreasingOrder(hashTable1, hashTable2, hashTable3, hashTable4, i):
	dataStream = createDataStream()
	minFreq9050 = None 
	heavyHittersClub = []

	for numFrequencyPair in dataStream:
		x = numFrequencyPair[0]
		frequency = numFrequencyPair[1]
		countFrequencies = []

		for j in range(4):
			newX = str(x) + str(i-1)
			hexHash = hashlib.md5(newX).hexdigest()

			byteArray = bytearray.fromhex(hexHash)
			incrementSlot = byteArray[j+1]

			if (j==0):
				countFrequencies.append(hashTable1[incrementSlot])  
			elif(j == 1):
				countFrequencies.append(hashTable2[incrementSlot])

			elif(j == 3):
				countFrequencies.append(hashTable4[incrementSlot])
			elif(j == 2):
				countFrequencies.append(hashTable3[incrementSlot])
      		
        minFrequency = min(countFrequencies)
        if(x == 9050):
        	minFreq9050 = minFrequency
        if (isHeavyHitter(minFrequency)):
        	heavyHittersClub.append(x)
   	return minFreq9050, heavyHittersClub


def countMinSketch(i):
	dataStream = createDataStream()
	hashTable1 = [0]*256
	hashTable2 = [0]*256
	hashTable3 = [0]*256
	hashTable4 = [0]*256
	for numFrequencyPair in dataStream:
		x = numFrequencyPair[0]
		frequency = numFrequencyPair[1]
		for inc in range(int(frequency)):
			currentTableFrequencies = []
			for j in range(4):
				newX = str(x) + str(i-1)
				hexHash = hashlib.md5(newX).hexdigest()

				byteArray = bytearray.fromhex(hexHash)
				incrementSlot = byteArray[j+1]

				if (j==0):
					currentTableFrequencies.append(hashTable1[incrementSlot])  
				elif(j == 1):
					currentTableFrequencies.append(hashTable2[incrementSlot])
				elif(j == 2):
					currentTableFrequencies.append(hashTable3[incrementSlot])
				elif(j == 3):
					currentTableFrequencies.append(hashTable4[incrementSlot])
        	lowestCurrentCount = min(currentTableFrequencies)
        	updateIndices = []
        	for y in range(4):
        		if(currentTableFrequencies[y]) == lowestCurrentCount:
        			updateIndices.append(y)
        	for num in updateIndices:
        		if (num == 0):
        			hashTable1[incrementSlot] += 1
        		elif(num == 1):
        			hashTable2[incrementSlot] += 1
        		elif(num == 2):
        			hashTable3[incrementSlot] += 1
        		elif(num == 3):
        			hashTable4[incrementSlot] += 1
	return nonDecreasingOrder(hashTable1, hashTable2, hashTable3, hashTable4, i)


def callerFunction():
  #9050
  lastNumFrequencyGuess = []
  numHeavyHitters = []
  for x in range(1, 11):
    minFreq9050, heavyHittersClub = countMinSketch(x)
    lastNumFrequencyGuess.append(minFreq9050)
    numHeavyHitters.append(len(set(heavyHittersClub)))
    
  #print the value averaged over the 10 trials
  print "Non Decreasing Order:"
  print sum(lastNumFrequencyGuess) / float(len(lastNumFrequencyGuess))
  print sum(numHeavyHitters) / float(len(numHeavyHitters))


callerFunction()

