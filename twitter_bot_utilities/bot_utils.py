import tweepy


def get_twitter_api():
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    return tweepy.API(auth)