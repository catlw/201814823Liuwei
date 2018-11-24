from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def main():
    categories = ['comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'misc.forsale',
    'talk.politics.misc',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.religion.misc',
    'alt.atheism',
    'soc.religion.christian'];
    

    newsgroup_train = fetch_20newsgroups(subset = 'train',categories = categories);
    newsgroups_test = fetch_20newsgroups(subset = 'test',
                                        categories = categories);


    vectorizer = HashingVectorizer(stop_words = 'english',non_negative = True,
                                n_features = 10000)
    fea_train = vectorizer.fit_transform(newsgroup_train.data)
    fea_test = vectorizer.fit_transform(newsgroups_test.data);
    #create the Multinomial Naive Bayesian Classifier
    clf = MultinomialNB(alpha = 0.01) 
    clf.fit(fea_train,newsgroup_train.target);
    pred = clf.predict(fea_test);
    calculate_result(newsgroups_test.target,pred);

def calculate_result(actual,pred):
    m_precision = metrics.precision_score(actual,pred,average="weighted");
    print "precision"
    print m_precision

if __name__ == "__main__":
    main()
