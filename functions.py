# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:36:21 2020

@author: Zain
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop = stopwords.words('english')
nltk.download('punkt')
stemming = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
global dataset
words = set(nltk.corpus.words.words())

"""
Clearing Corpus by 
* Removing Stop words, 
* Applying Lemmatizing, 
* Applying Stemming

Steps: First we covert the courpus to all lower case, then tokenise and finally the apply 
the NLTK library for the given above functions.
In return we have final dataset and all cleared corpus.
"""
def identify_tokens(row):
    review = row['TextWithOutStopWords']
    tokens = nltk.word_tokenize(review)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

def stem_list(row):
    my_list = row['TextWithOutStopWords']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

def lemmatize_text(row):
    return [lemmatizer.lemmatize(w) for w in row]

def clear(dataset):
    
    dataset = pd.read_csv('Dataset/Reviews.csv')#,nrows=1000)
    dataset = dataset.sample(frac=0.2)
    dataset = dataset.iloc[:,[1,2,3,9]]
    dataset['TextWithOutStopWords'] = dataset['Text'].str.lower()
    #Tokenization
    dataset['TextWithOutStopWords'] = dataset.apply(identify_tokens,axis=1)
    #Stemming
    dataset['TextWithOutStopWords'] = dataset.apply(stem_list, axis=1)
    #Lemmenization
    dataset['TextWithOutStopWords'] = dataset['TextWithOutStopWords'].apply(lemmatize_text)
    #Removing Stop words
    dataset['TextWithOutStopWords'] = dataset['TextWithOutStopWords'].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))
    return dataset


#Random Center points according to given number clusters
def getCentroids(dataset,features,example,K):
    """
    "randomCent" will the shape of given features (columns) in the dataset.
    In the given below function we will select any random rows from all dataset,
    using Random.choice, and will return it.
    
    This will select K random points in the dataset and now they are K different
    clusters in the dataset.
    """
    Centroids = np.array([]).reshape(features,0) 
    for i in range(K):
        rand=random.randint(0,example-1)
        Centroids=np.c_[Centroids,dataset[rand]]
    return Centroids
    
def findDistance(dataset,rows,clusters,randomCent):
    """
    To find the Euclidian Distance
    """
    dist = np.array([]).reshape(rows,0)
    for i in range(clusters):
        tempDist=np.sum((dataset-randomCent[:,i])**2,axis=1)
        dist = np.c_[dist,tempDist]
    C = np.argmin(dist,axis=1)+1
    return dist,C

def findintertiaScore(clusters,final,randomCent):
    wcss = 0
    for k in range(clusters):
        wcss+=np.sum((final[k+1]-randomCent[:,k])**2)
    return wcss

def kmeansRandom(dataset,clusters):
    """
    * m is the number of training examples is the Dataset.
    * n is the number of features (columns) in the Dataset.
    * randomCent are the clusters no of randomly selected centroids
      in the dataset now.
    * final is the dictionary which will contains all the clusters and their
      containing examples.
    * Now after selecting the Random Centroids we have to find the distance 
      between each example of the dataset from the randomly selected centriods.
    * And after finding distance from centroid to each example in the dataset,
      now its time to add the closest distanced example to the specific centroid.
    * The function for finding distance we have used widely used concept "Equilidian Distance"
      SUM((Dataset - Centroid) ^ 2)
    * and we will repeat these step for 100 iterations, To find if its converged we can
      also use if the final dictionary is still changing for not but for this example for loop
      will work Perfect!!!
    * To Evaluate the K-means we will find the WCSS score which is basically a 
      "Within Cluster Sum of Squares" sum of square of the distance of each data point
      in all clusters to their respective centroids.
    * Return values: Final dictionary of clusters of Dataset and score for the K no cluster.
    """
    m = len(dataset)
    n = len(dataset.T)
    randomCent = getCentroids(dataset,n,m,clusters)
    final = {}
    for i in range(100):
        Y={}
        EuclidianDistance=np.array([]).reshape(m,0)
        for i in range(clusters):
            tempDist=np.sum((dataset-randomCent[:,i])**2,axis=1)
            EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        C = np.argmin(EuclidianDistance,axis=1)+1
        for k in range(clusters):
            Y[k+1]=np.array([]).reshape(n,0)
        for i in range(m):
            Y[C[i]]=np.c_[Y[C[i]],dataset[i]]
        for k in range(clusters):
            Y[k+1]=Y[k+1].T
        for k in range(clusters):
            randomCent[:,k]=np.mean(Y[k+1],axis=0)
        final=Y
        
    score = findintertiaScore(clusters,final,randomCent)
    return final,score


#Word-cloud finding similar
def create_word_cloud(string):
    maskArray = npy.array(Image.open("mask.jpg"))
    cloud = WordCloud(background_color = "white", max_words = 200, mask = maskArray, stopwords = set(STOPWORDS))
    cloud.generate(string)
    #cloud.to_file("wordCloud.png")
    plt.figure(figsize=(15,15))
    plt.imshow(cloud)