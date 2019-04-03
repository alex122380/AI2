import tweepy
import numpy as np
from dicgen import *

# keys and tokens from the Twitter Dev Console
consumer_key = 'Exmf1lXUqO89Sm1tIk62CH8mc'
consumer_secret = 'Q6GLPLRmkX1XOljSl02Zb2x0YSdXxP28D2L3Wkw1e6RUgxCovu'
access_token = '1111728489164615682-JVqhenKoMkuvsNM60m9vJqHTOc8YbV'
access_token_secret = 'W8W0aKrrZ6i9N619jh4nJtXbFzS4150QmT01bMonpZNXT'

# Function to extract tweets 
def get_tweets(username): 
          
    # Authorization to consumer key and consumer secret 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 

    # Access to user's access key and access secret 
    auth.set_access_token(access_token, access_token_secret) 

    # Calling api 
    api = tweepy.API(auth) 

    # 20 tweets to be extracted 
    number_of_tweets=20
    #tweets = api.user_timeline(screen_name=username,count=number_of_tweets) 
    tweets = api.user_timeline(screen_name=username,count=number_of_tweets, tweet_mode='extended')

    # Empty Array 
    tmp=[]

    # create array of tweet information: username,  
    # tweet id, date/time, text 
    tweets_for_csv = [tweet.full_text for tweet in tweets] # CSV file created  
    for j in tweets_for_csv: 

        # Appending tweets to the empty array tmp 
        tmp.append([j,sentimentdeg(j)])  

    # Returning the tweets 
    return(tmp) 

# Function to extract tweets for usernames in a file
def get_user_tweets():
	usertweets=[]
	with open('usernames.txt') as f:
		for username in f:
			#usertweets.append((username[0],username[1]))
			usertweets.append(get_tweets(username))
	return usertweets

