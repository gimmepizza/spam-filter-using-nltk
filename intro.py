import pickle
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk import word_tokenize, WordNetLemmatizer
from collections import Counter
import random
from nltk import NaiveBayesClassifier, classify


rootdir = "/home/gimmepizza/spam filter"
lemmatizer = WordNetLemmatizer()
stop_words = (stopwords.words('english'))
stop_words.append('-')
stop_words.append('.')
stop_words.append(',')
stop_words.append(':')
stop_words.append("subject")
 
# for directories, subdirs, files in os.walk(rootdir):
#    print(directories, subdirs, len(files))# Same as before, but only print the ham and spam folders
# print "yoooooooooooo\n"
# for directories, subdirs, files in os.walk(rootdir):
#     if (os.path.split(directories)[1]  == 'ham'):
#         print(directories, subdirs, len(files))
    
#     if (os.path.split(directories)[1]  == 'spam'):
#         print(directories, subdirs, len(files))
def loadtr (rootdir1):
	tr_emails=[]
	for file1 in os.listdir(rootdir1 +"/spam")[:len(os.listdir(rootdir1 +"/spam"))]:
		f = open(rootdir1 +"/spam/" + file1, 'r')
		tr_emails.append((f.read(),"spam"))
	for file1 in os.listdir(rootdir1 +"/ham")[:len(os.listdir(rootdir1 +"/ham"))]:
		f = open(rootdir1 +"/ham/" + file1, 'r')
		tr_emails.append((f.read(),"ham"))
	f.close()
	return tr_emails

def data(file1):
	# a_list = []
	# b_list=[]
	# for i in range(len(a_list)):
	# 	b_list.append(word_tokenize(unicode(a_list[i], errors='ignore')))
	# a_list=[]
	# filtered_sentence=[]
	# for i in range(len(b_list)):
	# 	a_list.append([lemmatizer.lemmatize(word.lower()) for word in b_list[i]])
	# 	filtered_sentence.append([w for w in a_list[i] if (not w in stop_words) and wordnet.synsets(w)])
	file1=(word_tokenize(unicode(file1, errors='ignore')))
	
	file1=([lemmatizer.lemmatize(word.lower()) for word in file1])
	
	file2=dict([(w,True) for w in file1 if (not w in stop_words) and wordnet.synsets(w)])
	
	return file2

tr_emails = loadtr(rootdir+"/enron1")
tr_emails += loadtr(rootdir+"/enron2")
tr_emails += loadtr(rootdir+"/enron3")
tr_emails += loadtr(rootdir+"/enron4")
tr_emails += loadtr(rootdir+"/enron5")
print "Data loaded"
random.shuffle(tr_emails)
print " Data shuffled"
features = [(data(email), label) for (email, label) in tr_emails]
print "Features Extracted"
train1=features[:int(len(features)*0.9)]
test1=features[int(len(features)*0.9):]

classifier = nltk.NaiveBayesClassifier.train(train1)

print "Classified the Data"
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

print ('Accuracy on the training set = ' + str(nltk.classify.accuracy(classifier, train1)))
print ('Accuracy of the test set = ' + str(nltk.classify.accuracy(classifier, test1)))




# TO READ THE PICKLE
# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()


# spam_dat=os.listdir(rootdir)
# spam_dat=data(spam_dat)
# rootdir = "/home/gimmepizza/spam filter/enron1/ham/"
# ham_dat=os.listdir(rootdir)
# ham_dat=data(ham_dat)
# features=[]
# print spam_dat
# print "\n\n\n\n"
# print ham_dat

# for i in range(len(spam_dat)):
# 	features[i].append(word: count for word, count in Counter(spam_dat[i]).items())

# for i in range(len(ham_dat)):
# 	features[i].append(word: count for word, count in Counter(ham_dat[i]).items())
# print features

# file_list = os.listdir(rootdir)
# print mostfreqwords(file_list)[:10]