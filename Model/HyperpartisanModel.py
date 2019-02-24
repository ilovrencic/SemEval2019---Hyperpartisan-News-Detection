import os
import getopt
import sys
import xml.sax
import lxml.sax
import lxml.etree
import re
import json
import numpy
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import tree, preprocessing
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
import gensim
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from urllib.parse import urlparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

ps = PorterStemmer()
stopWords = list(map(lambda x: ps.stem(x), stopwords.words('english')))
websites = []

with open('websites.txt') as f:
    for website in f:
        websites.append(website.strip().lower())

########## ARTICLE HANDLING ##########
def handleArticle(article, model, X):
    text = lxml.etree.tostring(article, method='text', encoding='UTF-8').decode()
    htmltext = lxml.etree.tostring(article, method='html', encoding='UTF-8').decode()
    tokens = nltk.word_tokenize(text)

    finalArray = numpy.zeros(300)
    cnt = 0
    cnt2 = 0

    triggerWords = ['trigger', 'triggered', 'triggering', 'triggers', 'fuck', 'fucking', 'fuckery', 'fucked']
    triggerCount = 0

    #get word2vec representation for every word in article
    for token in tokens:
        if token in triggerWords:
            triggerCount += 1  
        if token in model and token not in stopWords:
            finalArray += model[token] 
            cnt += 1
        else:
            cnt2 += 1

    if cnt != 0:
        finalArray /= cnt

    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)

    #get core part of url, and check if it is in websites list
    websiteMatchCount = 0
    urls = re.findall(r'(?<=<a href=")[^"]*', htmltext)
    for url in urls:
        site = urlparse(url).netloc
        if site.startswith('www.'):
            site = site[4:]
        if site.lower() in websites:
            websiteMatchCount += 1

    #get date of article publification
    dateParts = str(article.get('published-at')).split('-')
    date = 0
    if len(dateParts) >= 2:
        year = int(str(article.get('published-at')).split('-')[0])
        month = int(str(article.get('published-at')).split('-')[1])
        date = year * 12 + month


    finalArray = numpy.append(finalArray, ss['neg'])
    finalArray = numpy.append(finalArray, ss['neu'])
    finalArray = numpy.append(finalArray, ss['pos'])
    finalArray = numpy.append(finalArray, ss['compound'])

    finalArray = numpy.append(finalArray, date)

    finalArray = numpy.append(finalArray, websiteMatchCount)

    finalArray = numpy.append(finalArray, triggerCount)

    #progress print
    print(article.get('id'), end='\r')
    
    X[article.get('id')] = list(finalArray)

########## SAX FOR STREAM PARSING ##########
class HyperpartisanArticleExtractor(xml.sax.ContentHandler):
    def __init__(self, model, X):
        xml.sax.ContentHandler.__init__(self)
        self.lxmlhandler = "undefined"
        self.model = model
        self.X = X

    def startElement(self, name, attrs):
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()

            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                # pass to handleArticle function
                handleArticle(self.lxmlhandler.etree.getroot(), self.model, self.X)
                self.lxmlhandler = "undefined"        

class HyperpartisanTruthExtractor(xml.sax.ContentHandler):
    def __init__(self, X):
        xml.sax.ContentHandler.__init__(self)
        self.lxmlhandler = "undefined"
        self.X = X

    def startElement(self, name, attrs):
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()

            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                self.X[self.lxmlhandler.etree.getroot().get("id")] = 1 if self.lxmlhandler.etree.getroot().get('hyperpartisan') == 'true' else 0
                self.lxmlhandler = "undefined"

def write(model, inputDataset, inputTruth, testDataset, testTruth, outputDataset, outputTruth, outputTestDataset, outputTestTruth):
    start_time = time.time()
    
    X1 = {}
    with open(inputDataset) as inputRunFile:
        print('parsing training dataset')
        xml.sax.parse(inputRunFile, HyperpartisanArticleExtractor(model, X1))
    json.dump(X1, open(outputDataset, 'w'))

    Y1 = {}
    with open(inputTruth) as inputTruthFile:
        print('parsing training truth')
        xml.sax.parse(inputTruthFile, HyperpartisanTruthExtractor(Y1))
    json.dump(Y1, open(outputTruth, 'w'))

    X2 = {}
    with open(testDataset) as testDatasetFile:
        print('parsing validation dataset')
        xml.sax.parse(testDatasetFile, HyperpartisanArticleExtractor(model, X2))
    json.dump(X2, open(outputTestDataset, 'w'))

    Y2 = {}
    with open(testTruth) as testTruthFile:
        print('parsing validation truth')
        xml.sax.parse(testTruthFile, HyperpartisanTruthExtractor(Y2))
    json.dump(Y2, open(outputTestTruth, 'w'))

    elapsed_time = time.time() - start_time
    
    print('FINISHED!', elapsed_time)
    return X1, Y1, X2, Y2

def read(trainingDataset, trainingTruth, validationDataset, validationTruth):
    start_time = time.time()
    # print('reading X1...')
    X1 = json.load(open(trainingDataset, 'r'))
    # print('reading Y1...')
    Y1 = json.load(open(trainingTruth, 'r'))
    # print('reading X2...')
    X2 = json.load(open(validationDataset, 'r'))
    # print('reading Y2...')
    Y2 = json.load(open(validationTruth, 'r'))

    print('done!', time.time() - start_time)
    
    return X1, Y1, X2, Y2         

def fitFunc(clf, X1, Y1, X2, Y2):
    start_time = time.time()
    
    X1 = list(X1.values())
    Y1 = list(Y1.values())
    X2 = list(X2.values())
    Y2 = list(Y2.values())

    # print('scaling dataset..')
    X1 = preprocessing.scale(X1)
    X2 = preprocessing.scale(X2)

    print('***********************')
    print('starting fit with:', clf)
    clf = clf.fit(X1, Y1)
    #dump the model
    pickle.dump(clf, open('model', 'wb'))

    #validate
    predicted = clf.predict(X2)
    print('accuracy_score', accuracy_score(predicted, Y2))

    print('time elapsed:', time.time() - start_time)
    print('***********************')

    return(accuracy_score(predicted, Y2))
    
if __name__ == '__main__':
    print('loading word2vec model...')
    model = gensim.models.KeyedVectors.load_word2vec_format('../PPP/W2V/GoogleNews-vectors-negative300.bin', binary=True)

    param_grid = [{'C': [2**x for x in range(-5,5)]}]
    h = GridSearchCV(SVC(gamma='auto'), param_grid)

    classifiers = [h]
    sums = [0 for x in classifiers]

    #input dataset
    inputDataset = 'path/to/training/articles'
    inputTruth = 'path/to/training/truth'
    testDataset = 'path/to/validation/articles'
    testTruth = 'path/to/validation/truth'

    #output files to which the parsed feature vectors will be output
    outputDataset = s + '.txt'
    outputTruth = s + '-truth.txt'
    outputTestDataset = str(i) + '.txt'
    outputTestTruth = str(i) + '-truth.txt'

    #this will parse the dataset and save the feature vectors to files, for later use
    X1, Y1, X2, Y2 = write(model, inputDataset, inputTruth, testDataset, testTruth, outputDataset, outputTruth, outputTestDataset, outputTestTruth)

    #this reads the given files which were output in the previous method, used when testing different classifiers, without changing feature vectors
    # X1, Y1, X2, Y2 = read(outputDataset, outputTruth, outputTestDataset, outputTestTruth)

    #trains the model and validates using parsed feature vectors
    fitFunc(classifier, X1, Y1, X2, Y2)
