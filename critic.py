import json
import math
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
#42706342

text = ['Hop on pop', 'Hop off pop', 'Hop Hop hop']
print ("Original text is\n", '\n'.join(text))

vectorizer = CountVectorizer(min_df=0)

# call `fit` to build the vocabulary
vectorizer.fit(text)

# call `transform` to convert text to a bag of words
x = vectorizer.transform(text)

# CountVectorizer uses a sparse array to save memory, but it's easier in this assignment to 
# convert back to a "normal" numpy array
x = x.toarray()

print()
print ("Transformed text vector is \n", x)

# `get_feature_names` tracks which word is associated with each column of the transformed x
print()
print ("Words for each feature:")
print (vectorizer.get_feature_names())


def _imdb_review(rtid):
    """
    Query the RT reviews API, to return the first page of reviews 
    for a movie specified by its IMDB ID
    
    Returns a list of dicts
    """    
    #filename='%d.txt'%(rtid)
    data=json.loads(open('%d.txt'%(rtid),'r').read())
    data = data['reviews']
    data = [dict(fresh=r['freshness'], 
                 quote=r['quote'], 
                 critic=r['critic'],
                 publication=r['publication'],
                 rtid=rtid
                 ) for r in data]
    return data

def fetch_reviews(movies,row):
    m = movies.iloc[row]
    try:
        result = pd.DataFrame(_imdb_review(m['id']))
        result['title'] = m['title']
    except KeyError:
        return None
    return result

def build_table(movies, rows):
    dfs = [fetch_reviews(movies, r) for r in range(rows)]
    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs, ignore_index=True)


from io import StringIO  
movie_txt = requests.get('https://raw.github.com/cs109/cs109_data/master/movies.dat').text
movie_file = StringIO(movie_txt) # treat a string like a file
movies = pd.read_csv(movie_file, delimiter='\t')

#for r in range(20):
#    print(movies[['id', 'title','imdbID', 'year']].iloc[r])
#    print()

critics = build_table(movies,20)
#print(critics)
critics.to_csv('critics.csv', index=False)
critics = pd.read_csv('critics.csv')
df = critics.copy()
df['fresh'] = df.fresh == 'fresh'
grp = df.groupby('critic')
counts = grp.critic.count()  # number of reviews by each critic
means = grp.fresh.mean()  

def make_xy(critics, vectorizer=None):
    #Your code here    
    if vectorizer is None:
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(critics.quote)
#    print(X.toarray())
#    print(vectorizer.get_feature_names())
    X = X.tocsc()  # some versions of sklearn return COO format
    Y = (critics.fresh == 'fresh').values.astype(np.int)
    return X, Y

X, Y = make_xy(critics)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
clf = MultinomialNB().fit(xtrain, ytrain)

print ("Accuracy: %0.2f%%" % (100 * clf.score(xtest, ytest)))

training_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)
#
#print ("Accuracy on training data: %0.2f" % (training_accuracy))
#print ("Accuracy on test data:     %0.2f" % (test_accuracy))

def calibration_plot(clf, xtest, ytest):
    prob = clf.predict_proba(xtest)[:, 1]
    outcome = ytest
    data = pd.DataFrame(dict(prob=prob, outcome=outcome))
    
#    print(data)
    #group outcomes into bins of similar probability
    bins = np.linspace(0, 1,5)
    cuts = pd.cut(prob, bins)
    binwidth = bins[1] - bins[0]

    #freshness ratio and number of examples in each bin
    cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
    cal['pmid'] = (bins[:-1] + bins[1:]) / 2
    cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])
#    print(cal)
        
    #the calibration plot
    plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    plt.errorbar(cal.pmid, cal['mean'], cal['sig'])
    plt.plot(cal.pmid, cal.pmid, linestyle='--', lw=1, color='k')
    plt.ylabel("Empirical P(Fresh)")
    
    #the distribution of P(fresh)
    plt.subplot2grid((3, 1), (2, 0))  
    plt.bar(left=cal.pmid, height=cal['count'],
            width=0.95 * (binwidth))
#    
    plt.xlabel("Predicted P(Fresh)")
    plt.ylabel("Number")
    
calibration_plot(clf, xtest, ytest)

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    print(prob)
    rotten = y == 0
    fresh = ~rotten
    return prob[rotten, 0].sum() + prob[fresh, 1].sum()

from sklearn.cross_validation import KFold

def cv_score(clf, x, y, score_func):
    result = 0
    nfold = 5
    for train, test in KFold(y.size, nfold): # split data into train/test groups, 5 times
        clf.fit(x[train], y[train]) # fit
        result += score_func(clf, x[test], y[test]) # evaluate score function on held-out data
    return result / nfold # average

#alphas = [0, .1, 1, 5, 10, 50]
alphas = [0, .1, 1, 5]
min_dfs = [1e-5, 1e-4, 1e-3]

#Find the best value for alpha and min_df, and the best classifier
best_alpha = None
best_min_df = None
max_loglike = -np.inf

for alpha in alphas:
    for min_df in min_dfs:         
        vectorizer = CountVectorizer(min_df = min_df)       
        X, Y = make_xy(critics, vectorizer)
        
        #your code here
        clf = MultinomialNB(alpha=alpha)
        loglike = cv_score(clf, X, Y, log_likelihood)

        if loglike > max_loglike:
            max_loglike = loglike
            best_alpha, best_min_df = alpha, min_df

#print ("alpha: %f" % best_alpha)
#print ("min_df: %f" % best_min_df)

vectorizer = CountVectorizer(min_df = best_min_df)
X, Y = make_xy(critics, vectorizer)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

#from sklearn.feature_extraction.text import TfidfTransformer
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(xtrain)
#X_train_tfidf.shape

clf = MultinomialNB(alpha=best_alpha).fit(xtrain, ytrain)

calibration_plot(clf, xtest, ytest)

# Your code here. Print the accuracy on the test and training dataset
training_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)

#print ("Accuracy on training data: %0.2f" % (training_accuracy))
#print ("Accuracy on test data:     %0.2f" % (test_accuracy))

#words = np.array(vectorizer.get_feature_names())
#
#x = np.eye(xtest.shape[1])
#probs = clf.predict_log_proba(x)[:, 0]
#ind = np.argsort(probs)
#
#good_words = words[ind[:10]]
#bad_words = words[ind[-10:]]
#
#good_prob = probs[ind[:10]]
#bad_prob = probs[ind[-10:]]
#
#print ("Good words\t     P(fresh | word)")
#for w, p in zip(good_words, good_prob):
#    print ("%20s" % w, "%0.2f" % (1 - np.exp(p)))
#    
#print ("Bad words\t     P(fresh | word)")
#for w, p in zip(bad_words, bad_prob):
#    print ("%20s" % w, "%0.2f" % (1 - np.exp(p)))
#
##Your code here
#x, y = make_xy(critics, vectorizer)
#
#prob = clf.predict_proba(x)[:, 0]
#predict = clf.predict(x)
#
#bad_rotten = np.argsort(prob[y == 0])[:5]
#bad_fresh = np.argsort(prob[y == 1])[-5:]
#
#print ("Mis-predicted Rotten quotes")
#print ('---------------------------')
#for row in bad_rotten:
#    print (critics[y == 0].quote.iloc[row])
#    print()
#
#print ("Mis-predicted Fresh quotes")
#print ('--------------------------')
#for row in bad_fresh:
#    print (critics[y == 1].quote.iloc[row])
#    print()
#
#print(clf.predict_proba(vectorizer.transform(['This movie is not remarkable, touching, or superb in any way'])))
##print(clf.predict_proba(vectorizer.transform(['This movie is not bad and gloomy '])))


