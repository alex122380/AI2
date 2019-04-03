import sys
import numpy as np
# tweets=[]

# with open('datafile.txt') as f:
#     for line in f:
#         row = line.rstrip().split('\t')
#         tweets.append([int(row[0]), row[1]])
# for i in tweets[:,1]:
# 	i=i.split(' ')
# 	for j in i:
# 		j = lower(j)

tweets=np.load('data.npy')
for tweet in tweets:
	tweet[3]=tweet[3].strip()
onecount=0.0
posmap={}
negmap={}
for tweet in tweets:
	#print(tweet[1])
	onecount+=float(tweet[1])
	wordset=set(tweet[3])
	if float(tweet[1])==0:
		for word in wordset:
			if word not in negmap:
				negmap[word]=0
			negmap[word]+=1
	elif float(tweet[1])==0.5:
		for word in wordset:
			if word not in negmap:
				negmap[word]=0
			if word not in posmap:
				posmap[word]=0
			negmap[word]+=0.5
			posmap[word]+=0.5
	else:
		for word in wordset:
			if word not in posmap:
				posmap[word]=0
			posmap[word]+=1
prioroneprob=onecount/np.size(tweets,0)
priorzeroprob=1-prioroneprob
#print(prioroneprob)
#print(priorzeroprob)
def oneprob(word) -> float:
	if word not in posmap:
		return sys.float_info.min
	return posmap[word]/onecount
def zerprob(word) -> float:
	if word not in negmap:
		return sys.float_info.min
	return negmap[word]/(np.size(tweets,0)-onecount)

def sentimentdeg(tweet) -> float:
	oneoverzero=prioroneprob/priorzeroprob
	for word in set(tweet):
		oneoverzero*=oneprob(word)/zerprob(word)
	return oneoverzero
