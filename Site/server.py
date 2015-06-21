import tornado.ioloop
import tornado.web
import os
import pandas as pd
import numpy as np
import nltk
import re
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer



class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class BubbleData(tornado.web.RequestHandler):
    def get(self):
        df = self.df

        data = df.groupby([lambda x: df['Date2'][x].year,lambda y: df['Time2'][y].hour,"Location"])["Fatalities"].sum().order(ascending=False)
        dataf = pd.DataFrame(data)
        dataf = dataf.reset_index()
        data = dataf.rename(columns={'level_0':'x','level_1':'y','Fatalities':'z'})
        data = data.loc[data["z"]>100]
        data = data.to_dict("records")
        self.write({"data" : data})

    def initialize(self, df):
        self.df = df

class BubbleDetailedData(tornado.web.RequestHandler):
    def get(self):
        df = self.df
        location = self.get_argument("location")

        detail = df.loc[df["Location"] == location]
        dtf = detail.groupby("Operator").count().apply(lambda x: 100*x/float(x.sum()))
        dtf = dtf.reset_index()
        data = dtf[["Operator","Date"]]
        data = data.rename(columns={'Date':'y'})

        data = data.to_dict("records")

        del detail["Date2"]
        del detail["Time2"]
        del detail["Flight #"]

        detail.columns = map(str.lower, detail.columns)
        detail = detail.to_dict("records")
        self.write({"data" : data, "complete":detail})

    def initialize(self, df):
        self.df = df

settings = {"template_path" : os.path.dirname(__file__),
            "static_path" : os.path.join(os.path.dirname(__file__),"static"),
            "debug" : True
            }

if __name__ == "__main__":

    path = os.path.join(os.path.dirname(__file__), "Airplane_Crashes_and_Fatalities_Since_1908.csv")
    print('loading...')
    df = pd.read_csv(path)
    df["Date2"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
    df["Time2"] = pd.to_datetime(df.Time, format="%H:%M",coerce=True)
    df["Summary"] = df["Summary"].fillna('no-reason')
    df["Route"] = df["Route"].fillna('no-route')
    df["Type"] = df["Type"].fillna('no-type')
    df["Registration"] = df["Registration"].fillna('no-registration')
    df["cn/In"] = df["cn/In"].fillna('no-cn/in')
    df["Aboard"] = df["Aboard"].fillna('-1')
    df["Fatalities"] = df["Fatalities"].fillna('-1')
    df["Ground"] = df["Ground"].fillna('-1')
    df["Time"] = df["Time"].fillna('')

    def summary_to_words( raw_summary ):
        #print(raw_summary)
        # 2. Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", raw_summary)
        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        #
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))
        #
        # 5. Remove stop words
        meaningful_words = [w for w in words if not w in stops]
        #
        # 6. Join the words back into one string separated by space,
        # and return the result.
        return( " ".join( meaningful_words ))


    num_summary = df["Summary"].size
    clean_train_summary = []


    for i in range( 0, num_summary ):
        if( (i+1)%2500 == 0 ):
            print ("Summary %d of %d\n" % ( i+1, num_summary ))
        # Call our function for each one, and add the result to the list
        clean_train_summary.append( summary_to_words( df["Summary"][i] ) )



    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_summary)

    train_data_features = train_data_features.toarray()
    print(train_data_features.shape)

    vocab = vectorizer.get_feature_names()
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    data_to_export = {"word": vocab,"count":dist}
    data = pd.DataFrame(data_to_export)



    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/bubbledata", BubbleData,{"df":df}),
        (r"/bubbledetaileddata", BubbleDetailedData,{"df":df}),
        (r"/static/(.*)", tornado.web.StaticFileHandler,
            {"path": settings["static_path"]})

    ], **settings)
    application.listen(8100)
    print("ready")
    tornado.ioloop.IOLoop.current().start()

