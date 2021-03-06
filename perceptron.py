import string
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
from dicgen import *
from downloadtweets import *

'''
# Decision function.
def h(w, x):
  dot = sum(pair[0] * pair[1] for pair in zip(w, x))
  if dot >= 0:
    return 1
  else:
    return 0
'''
# Decision function.
def h(w, x):
  dot = sum(pair[0] * pair[1] for pair in zip(w, x))
  if dot >= 0.5:
    return 1
  elif dot >= -0.5: # Between -0.5 to 0.5.
    return 0.5
  else:             # Below -0.5
    return 0

def calculateAccuracy(model, valuesList, labelList):
  errorCount = 0
  i = 0 # Loop counter.
  for example in valuesList:
    yPredict = h(model, example)
    if yPredict != labelList[i]:
      errorCount += 1
    i += 1
  accuracy = (len(valuesList) - errorCount) / len(valuesList)
  return accuracy

# Each elementList in userList is: [sentcal_1, sentcal_2, ..., sentcal_20]
userList = []
labelList = []
elementList = []

with open('labelList.txt') as labelFile:
  for line in labelFile:
    labelList.append(float(line))
print()

input = get_user_tweets()
for user in input:
  tempList = []
  for tweet in user:
    tempList.append(tweet[1]) # sentcal.
  userList.append([float(i) for i in tempList])

'''
# Assign every 20 tweets to a user, and randomly assign a ALRIGHT label to the user.
i = 0          # Counter for assigning sentcal to user.(Each user gets 20 sentcals.)
lineNumber = 0 # For debugging.
for tweet in tweets:
  print(lineNumber)
  lineNumber += 1
  sentcal = sentimentdeg(tweet[3])
  elementList.append(sentcal)
  i += 1
  if (i == 20): # Already assigned 20 sentcals to a user.
    #label = random.randint(0, 1) # Randomly generates an ALRIGHT label.
    #label = random.uniform(0, 1)
    #if label >= 0.75:
    #  label = 1
    #elif label >= 0.25:
    #  label = 0.5
    #else:
    #  label = 0
    #labelList.append(label)
    tempList = list(elementList)
    userList.append(tempList)
    elementList.clear() # Clear elementList.
    i = 0 # Reset counter.
'''


# Train the weights.
L = 100                                 # The number of training epochs.
i = 0                                  # Loop counter.
w = [0] * 20
trainAccuracyList = []
testAccuracyList = []

while (i < L):
  j = 0 # Loop counter for labelList.
  for user in userList:
    adjust = [(labelList[j] - h(w, user)) * element for element in user] # adjust = (y - hw(x))*x
    w = [sum(pair) for pair in zip(w, adjust)]                           # w = w + (y - hw(x))*x
    j += 1
    
  trainAccuracy = calculateAccuracy(w, userList, labelList)
  trainAccuracyList.append(trainAccuracy)
  
  print('epoch:', i)
  i += 1

print(trainAccuracyList)
plt.plot(list(range(100)), trainAccuracyList)
plt.show()












