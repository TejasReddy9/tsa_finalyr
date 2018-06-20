from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s 

#consumer key, consumer secret, access token, access secret.
ckey="gBXjfuano8qpIhHMrOJe12TJR"
csecret="azwXRAKcKs4MEmPx5CqGEWT8lebDu170d4jhS72xNTPyLBFES2"
atoken="72244363-bUgoBVDHLE8NYtnNXSXyUkh3oVh5UotX3IMR9xwbX"
asecret="oOYptak3J4EmzaK0MS5Hh6Gs9pxHTNjMbH03DETZE831Z"

class listener(StreamListener):
	def on_data(self, data):
		try:
			all_data = json.loads(data)
			tweet = all_data["text"]
			sentiment_value, confidence = s.sentiment(tweet)
			print(tweet, sentiment_value, confidence)

			if(confidence*100 >= 80):
				output = open("twitter-out.txt","a")
				output.write(sentiment_value)
				output.write('\n')
				output.close()
			return(True)
		except Exception as e:
			return(True)

	def on_error(self, status):
		print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])

