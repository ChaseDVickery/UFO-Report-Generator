
import os
import tweepy

envvars = os.environ

def get_twitter_client():
    consumer_key = envvars.get('TWITTER_API_KEY_UFO')
    consumer_secret = envvars.get('TWITTER_API_SECRET_UFO')
    access_token = envvars.get('TWITTER_ACCESS_TOKEN_UFO')
    access_token_secret = envvars.get('TWITTER_ACCESS_SECRET_UFO')
    client = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret)
    return client