import spacy
import pandas
import random
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

# load reviews data
dataframe = pandas.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
clean_data = dataframe.dropna(subset = ["reviews.text"])
reviews_data = clean_data["reviews.text"]
print(f"This dataset has {(len(reviews_data))} entries.")

# sentiment analysis function
def analyze_sentiment(sentence):
    doc = nlp(sentence)
    polarity = doc._.blob.polarity
    sentiment = doc._.blob.sentiment
    print(f"Analyzing sentence: {sentence}")
    print(sentiment)
    if(polarity > 0):
        print("The review is likely positive.")
    else:
        print("The review is likely negative.")
    print()

'''
Analyzing sentence: I order 3 of them and one of the item is bad quality. Is missing backup spring so I have to put a pcs of aluminum to make the battery work.
Sentiment(polarity=-0.44999999999999996, subjectivity=0.35833333333333334)
The review is likely negative.
'''

# analyze 10 random reviews
for i in range(0, 10):
    review_num = random.randint(0, len(reviews_data))
    print(f"Randomly selecting: review #{review_num}")
    analyze_sentiment(reviews_data[review_num])