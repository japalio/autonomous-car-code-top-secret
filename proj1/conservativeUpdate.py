import random
#testing
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
  	currentTableFrequencies = []

    for j in range(4):

      newX = str(x) + str(i-1)
      
      
      #Calculate the MD5 score of the resulting string
      

      hexHash = hashlib.md5(newX).hexdigest()
      
      #The hash value is the j-th byte of the score.
      #incrementSlot is the decimal value 
      byteArray = bytearray.fromhex(hexHash)
      incrementSlot = byteArray[j + 1]
      
      
  
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
    for x in range(4):
    	if(currentTableFrequencies[x] == lowestCurrentCount):
    		updateIndices.append(x)


   	for num in updateIndices:
   		if(num == 0):
   			hashTable1[incrementSlot] += frequency
   		elif(num == 1):
   			hashTable2[incrementSlot] += frequency
   		elif(num == 2):
   			hashTable3[incrementSlot] += frequency
   		elif(num == 3):
   			hashTable4[incrementSlot] += frequency

  return nonDecreasingOrder(hashTable1, hashTable2, hashTable3, hashTable4, i)
