import random
import math
import hashlib 
#testing

def isHeavyHitter(minFrequency):
  if minFrequency > 879:
    return True
  else:
    return False

def createDecreasingDataStream():
  dataStreamToReverse = createDataStream()
  dataStream = list(reversed(dataStreamToReverse))
  return dataStream

def createRandomDataStream():
  dataStreamTuples = createDataStream()
  dataStream = []
  for tup in dataStreamTuples:
    freq = tup[1]
    for i in range(int(freq)):
      dataStream.append(tup[0])


  random.shuffle(dataStream)
  return dataStream

    
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

def RandomOrder(hashTable1, hashTable2, hashTable3, hashTable4, i):
  dataStream = createRandomDataStream()
  minFreq9050 = None
  heavyHittersClub = []

  for data in dataStream:
    #for each of the 4 independent hash tables
    x = data
    countFrequencies = []

    for j in range(4):

      newX = str(x) + str(i-1)

      #Calculate the MD5 score of the resulting string

      hexHash = hashlib.md5(newX).hexdigest()

      #The hash value is the j-th byte of the score.
      #incrementSlot is the decimal value 
      byteArray = bytearray.fromhex(hexHash)
      incrementSlot = byteArray[j + 1]


      #increment the count 
      if(j == 0):
        countFrequencies.append(hashTable1[incrementSlot])  
      elif(j == 1):
        countFrequencies.append(hashTable2[incrementSlot])
      elif(j == 2):
        countFrequencies.append(hashTable3[incrementSlot])
      elif(j == 3):
        countFrequencies.append(hashTable4[incrementSlot])

    minFrequency = min(countFrequencies)
    if(x == 9050):
      minFreq9050 = minFrequency
    if(isHeavyHitter(minFrequency)):
      heavyHittersClub.append(x)

  return minFreq9050, heavyHittersClub

def countMinSketchRandom(i):
  dataStream = createRandomDataStream()
  # print dataStream
  hashTable1 = [0]*256
  hashTable2 = [0]*256
  hashTable3 = [0]*256
  hashTable4 = [0]*256
  for data in dataStream:
    #for each of the 4 independent hash tables
    x = data 
    currentTableFrequencies = []
    incrementSlots = []
    for j in range(4):
      newX = str(x) + str(i-1)
      #Calculate the MD5 score of the resulting string
      hexHash = hashlib.md5(newX).hexdigest()
      
      #The hash value is the j-th byte of the score.
      #incrementSlot is the decimal value 
      byteArray = bytearray.fromhex(hexHash)
      incrementSlot = byteArray[j + 1]
      incrementSlots.append(incrementSlot)
      #increment the count 
      if(j == 0):
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
      if(num == 0):
        hashTable1[incrementSlots[0]] += 1
      elif(num == 1):
        hashTable2[incrementSlots[1]] += 1
      elif(num == 2):
        hashTable3[incrementSlots[2]] += 1
      elif(num ==3):
        hashTable4[incrementSlots[3]] += 1


  return RandomOrder(hashTable1, hashTable2, hashTable3, hashTable4, i)

def nonIncreasingOrder(hashTable1, hashTable2, hashTable3, hashTable4, i):
  dataStream = createDecreasingDataStream()
  minFreq9050 = None
  heavyHittersClub = []

  for numFrequencyPair in dataStream:
    #for each of the 4 independent hash tables
    x = numFrequencyPair[0]
    frequency = numFrequencyPair[1]
    countFrequencies = []
    for j in range(4):

      newX = str(x) + str(i-1)

      #Calculate the MD5 score of the resulting string

      hexHash = hashlib.md5(newX).hexdigest()

      #The hash value is the j-th byte of the score.
      #incrementSlot is the decimal value 
      byteArray = bytearray.fromhex(hexHash)
      incrementSlot = byteArray[j + 1]




      if(j == 0):
        countFrequencies.append(hashTable1[incrementSlot])  
      elif(j == 1):
        countFrequencies.append(hashTable2[incrementSlot])
      elif(j == 2):
        countFrequencies.append(hashTable3[incrementSlot])
      elif(j == 3):
        countFrequencies.append(hashTable4[incrementSlot])
    
    minFrequency = min(countFrequencies)
    #print countFrequencies
    if(x == 9050):
      minFreq9050 = minFrequency
    if(isHeavyHitter(minFrequency)):
      heavyHittersClub.append(x)

  return minFreq9050, heavyHittersClub
    
def nonDecreasingOrder(hashTable1, hashTable2, hashTable3, hashTable4, i):
  dataStream = createDataStream()
  minFreq9050 = None
  heavyHittersClub = []

  for numFrequencyPair in dataStream:
    #for each of the 4 independent hash tables
    x = numFrequencyPair[0]
    frequency = numFrequencyPair[1]
    countFrequencies = []
    for j in range(4):

      newX = str(x) + str(i-1)

      #Calculate the MD5 score of the resulting string

      hexHash = hashlib.md5(newX).hexdigest()

      #The hash value is the j-th byte of the score.
      #incrementSlot is the decimal value 
      byteArray = bytearray.fromhex(hexHash)
      incrementSlot = byteArray[j + 1]




      if(j == 0):
        countFrequencies.append(hashTable1[incrementSlot])  
      elif(j == 1):
        countFrequencies.append(hashTable2[incrementSlot])
      elif(j == 2):
        countFrequencies.append(hashTable3[incrementSlot])
      elif(j == 3):
        countFrequencies.append(hashTable4[incrementSlot])
    
    minFrequency = min(countFrequencies)
    #print countFrequencies
    if(x == 9050):
      minFreq9050 = minFrequency
    if(isHeavyHitter(minFrequency)):
      heavyHittersClub.append(x)

  return minFreq9050, heavyHittersClub

def countMinSketchDecreasing(i):
  dataStream = createDecreasingDataStream()
  # print dataStream
  hashTable1 = [0]*256
  hashTable2 = [0]*256
  hashTable3 = [0]*256
  hashTable4 = [0]*256
  count = 0
  for numFrequencyPair in dataStream:
    count += 1
    #for each of the 4 independent hash tables
    x = numFrequencyPair[0]
    frequency = numFrequencyPair[1]
    
    for inc in range(int(frequency)):
      currentTableFrequencies = []
      incrementSlots = []
      # print x
      for j in range(4):
        newX = str(x) + str(i-1)
        hexHash = hashlib.md5(newX).hexdigest()
        byteArray = bytearray.fromhex(hexHash)
        incrementSlot = byteArray[j + 1]
        incrementSlots.append(incrementSlot)  
    
        if(j == 0):
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
        if(currentTableFrequencies[y] == lowestCurrentCount):
          updateIndices.append(y)
      
      for num in updateIndices:
        if(num == 0):
          hashTable1[incrementSlots[0]] += 1
        elif(num == 1):
          hashTable2[incrementSlots[1]] += 1
        elif(num == 2):
          hashTable3[incrementSlots[2]] += 1
        elif(num == 3):
          hashTable4[incrementSlots[3]] += 1
      
  return nonIncreasingOrder(hashTable1, hashTable2, hashTable3, hashTable4, i)


def countMinSketch(i):
  dataStream = createDataStream()
  # print dataStream
  hashTable1 = [0]*256
  hashTable2 = [0]*256
  hashTable3 = [0]*256
  hashTable4 = [0]*256
  count = 0
  for numFrequencyPair in dataStream:
    count += 1
    #for each of the 4 independent hash tables
    x = numFrequencyPair[0]
    frequency = numFrequencyPair[1]

    
    for inc in range(int(frequency)):
      currentTableFrequencies = []
      incrementSlots = []
      # print x

      for j in range(4):

        newX = str(x) + str(i-1)
        
        
        #Calculate the MD5 score of the resulting string

        hexHash = hashlib.md5(newX).hexdigest()
        
        #The hash value is the j-th byte of the score.
        #incrementSlot is the decimal value 
        byteArray = bytearray.fromhex(hexHash)
        incrementSlot = byteArray[j + 1]
        incrementSlots.append(incrementSlot)
        
        
    
        if(j == 0):
          currentTableFrequencies.append(hashTable1[incrementSlot])
        elif(j == 1):
          currentTableFrequencies.append(hashTable2[incrementSlot])
        elif(j == 2):
          currentTableFrequencies.append(hashTable3[incrementSlot])
        elif(j == 3):
          currentTableFrequencies.append(hashTable4[incrementSlot])

        


      lowestCurrentCount = min(currentTableFrequencies)
      # print x
      # print frequency
      # print currentTableFrequencies
      updateIndices = []
      for y in range(4):
      	if(currentTableFrequencies[y] == lowestCurrentCount):
      		updateIndices.append(y)
      # print "current table frequencies ", currentTableFrequencies
      # print "hash table before:", hashTable1[incrementSlot], " ", hashTable2[incrementSlot], " ", hashTable3[incrementSlot], " ", hashTable4[incrementSlot]
      # print "lowest current count: ", lowestCurrentCount
      # print "update indices: ", updateIndices

      
      for num in updateIndices:
     		if(num == 0):
     			hashTable1[incrementSlots[0]] += 1
     		elif(num == 1):
     			hashTable2[incrementSlots[1]] += 1
     		elif(num == 2):
     			hashTable3[incrementSlots[2]] += 1
     		elif(num == 3):
     			hashTable4[incrementSlots[3]] += 1
      # print "hash table after:", hashTable1[incrementSlot], " ", hashTable2[incrementSlot], " ", hashTable3[incrementSlot], " ", hashTable4[incrementSlot]

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


  lastNumFrequencyGuess = []
  numHeavyHitters = []
  for x in range(1, 11):
    minFreq9050, heavyHittersClub = countMinSketchDecreasing(x)
    lastNumFrequencyGuess.append(minFreq9050)
    numHeavyHitters.append(len(set(heavyHittersClub)))
    
  #print the value averaged over the 10 trials
  print "Non Increasing Order:"
  print sum(lastNumFrequencyGuess) / float(len(lastNumFrequencyGuess))
  print sum(numHeavyHitters) / float(len(numHeavyHitters))



  lastNumFrequencyGuess = []
  numHeavyHitters = []
  for x in range(1, 11):
    minFreq9050, heavyHittersClub = countMinSketchRandom(x)
    lastNumFrequencyGuess.append(minFreq9050)
    numHeavyHitters.append(len(set(heavyHittersClub)))
  #print the value averaged over the 10 trials
  print "Random Order:"
  print sum(lastNumFrequencyGuess) / float(len(lastNumFrequencyGuess))
  print sum(numHeavyHitters) / float(len(numHeavyHitters))

  

callerFunction()
