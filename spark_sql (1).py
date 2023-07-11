#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:59:07 2019

@author: navyarao
"""
import pandas as pd
import numpy as np
from collections import namedtuple
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import roc_curve, auc, classification_report

from pyspark import SparkConf, SparkContext
import sys
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql import functions as f
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf

from collections import namedtuple
import matplotlib.pyplot as plt


conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")

sc = SparkContext()

sqlContext = SQLContext(sc)
df = sqlContext.read.csv("C:/Users/Akhilesh/Downloads/amazon-fine-food-reviews/Reviews.csv",header=True)
#EDA

import seaborn as sns
df.dtypes
df.printSchema
df.describe(["Score"]).show()

plot_score = plt.axes()
sns.countplot(df.Score,plot_score=plot_score)
plot_score.set_title('Score Distribution')
plt.show()

df.ix[df.Score>3,'Sentiment']="POSITIVE"
df.ix[df.Score<=3,'Sentiment']="NEGATIVE"

posneg = plt.axes()
sns.countplot(df.Sentiment,posneg=posneg)
posneg.set_title('Sentiment Positive vs Negative Distribution')
plt.show()

print("Percentage pf positive review:", len(df[df.Sentiment=="POSITIVE"])/len(df))
print("Percentage of positive review:",len(df[df.Sentiment=="NEGATIVE"])/len(df))

#Percentage of positive review: 0.7806735461444549
#Percentage of positive review: 0.21932645385554503

df.summary("count", "min", "25%", "75%", "max").show()
distinct = df.select("ProductId").distinct().show(n=20)

review_count = df.groupBy("Score").count().collect()

#sqlContext.sql("SELECT * FROM df WHERE SCORE != 3").show()




df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()


df.select([c for c in df.columns if c in ['Summary','Score']]).show()



df = df.withColumn(
    'Status',
    f.when((f.col("Score") >= 3), "Positive")\
    .otherwise("Negative")
)

df.show()
df.schema



#def cleanup_text(record):
#    text  = record[8]
#    uid   = record[9]
#    words = text.split()
#    
#    # Default list of Stopwords
#    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
#    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
#    u'can', 'cant', 'come', u'could', 'couldnt', 
#    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
#    u'each', 
#    u'few', 'finally', u'for', u'from', u'further', 
#    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
#    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
#    u'just', 
#    u'll', 
#    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
#    u'no', u'nor', u'not', u'now', 
#    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
#    u'r', u're', 
#    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
#    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
#    u'under', u'until', u'up', 
#    u'very', 
#    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
#    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves']
#    
#    # Custom List of Stopwords - Add your own here
#    stopwords_custom = ['']
#    stopwords = stopwords_core + stopwords_custom
#    stopwords = [word.lower() for word in stopwords]    
#    
#    text_out = [re.sub('[^a-zA-Z0-9]','',word) for word in words]                                       # Remove special characters
#    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
#    return text_out
# 


from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer





# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="Text", outputCol="words1", pattern="\\W")
# stop words
add_stopwords = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such',
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves']


stopwordsRemover = StopWordsRemover(inputCol="words1", outputCol="filtered").setStopWords(add_stopwords)

tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")
hashtf = HashingTF(numFeatures=2**16, inputCol="filtered", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms

# bag of words count

#hashtf = HashingTF(numFeatures=2**16, inputCol="tokens", outputCol='tf')
#
#idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
#
#


pipeline = Pipeline(stages=[regexTokenizer,stopwordsRemover,tokenizer,hashtf,idf])

pipelineFit= pipeline.fit(df)





train_df = pipelineFit.transform(df)
train_df.show()

#
## bag of words count
#countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
#
#from pyspark.ml import Pipeline
#from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
##label_stringIdx = StringIndexer(inputCol = "Category", outputCol = "label")
#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])
## Fit the pipeline to training documents.
#pipelineFit = pipeline.fit(df)
#dataset = pipelineFit.transform(df)
#dataset.show(5)



def tag_and_remove(data_str):
    cleaned_str = ' '
    # noun tags
    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags + jj_tags + vb_tags

    # break string into 'words'
    text = data_str.split()

    # tag the text and keep only those with the right tags
    tagged_text = pos_tag(text)
    for tagged_word in tagged_text:
        if tagged_word[1] in nltk_tags:
            cleaned_str += tagged_word[0] + ' '

    return cleaned_str






def lemmatize(data_str):
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    text = data_str.split()
    tagged_words = pos_tag(text)
    for word in tagged_words:
        if 'v' in word[1].lower():
            lemma = lmtzr.lemmatize(word[0], pos='v')
        else:
            lemma = lmtzr.lemmatize(word[0], pos='n')
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str


#!pip install preproc

#import preproc as pp

tag_and_remove_udf = udf(tag_and_remove, StringType())
lemmatize_udf = udf(lemmatize, StringType())







