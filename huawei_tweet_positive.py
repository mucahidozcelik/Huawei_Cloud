import snscrape.modules.twitter as sntwitter
import csv
import pandas as pd
import stopword as stopword
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


maxTweets = 10000

csvFile = open('positive.csv', 'a', newline='', encoding='utf8')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['text'])

for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper('#positive + since:2017-12-22 until:2022-12-22 lang:"en"').get_items()):
    if i >= maxTweets:
        break
    csvWriter.writerow([tweet.content])
    print(f"{i}/{maxTweets} ({round(i/maxTweets*100, 2)}%) tweets processed")
    #time.sleep(0.5)  # API rate limitlerini koruma amaçlı bir süre bekleyin
csvFile.close()


df = pd.read_csv('/Users/mucahidozcelik/Desktop/huaweiproject/positive.csv')
df.reset_index(inplace=True)

df.columns = ["index", "tweet"]
df.drop(columns=['index'], inplace=True)

df.head()
df.info

###############################
# Normalizing Case Folding
###############################
df2 = pd.DataFrame()

df2['tweet'] = df['tweet'].str.lower()
df2['tweet']

###############################
# Punctuations
###############################

df2['tweet'] = df2['tweet'].str.replace('[^\w\s]', '',regex=True)
df2['tweet']

###############################
# Numbers
###############################

df2['tweet'] = df2['tweet'].str.replace( "\d" , "" , regex=True)


###############################
# Stopwords
###############################
import nltk


def remove_stopwords(text, stopwords):
    filtered_tokens = [token for token in nltk.word_tokenize(text) if token.lower() not in stopwords]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text

# İngilizce stop words listesi
stopwords = nltk.corpus.stopwords.words("english")

df2['tweet'] = df2['tweet'].apply(remove_stopwords, stopwords=stopwords)
df2['tweet']


###############################
# Sentiment Analysis
###############################

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome")
sia.polarity_scores("I liked this music but it is not good as the other one")

df2["sentiment"] = df2["tweet"].apply(lambda x: "positive" if sia.polarity_scores(x)["compound"] > 0 else "negative" if sia.polarity_scores(x)["compound"] < 0 else "neutral")

df2["sentiment"].value_counts()
df2.head()

df2.to_csv("positive_tweets.csv")

