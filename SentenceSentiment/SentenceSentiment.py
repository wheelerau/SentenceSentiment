import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

data_dir = 'C:\\temp\\SentenceSentiment'
classes = ['pos', 'neg']

# Read the data
train_data = []
train_labels = []
test_data = []
test_labels = []
for curr_class in classes:
    dirname = os.path.join(data_dir, curr_class)

    for fname in os.listdir(dirname):
        with open(os.path.join(dirname, fname), 'r') as f:
            content = f.read()
            if fname.startswith('cv9'):
                test_data.append(content)
                test_labels.append(curr_class)
            else:
                train_data.append(content)
                train_labels.append(curr_class)

# Create feature vectors
# min_df=5, discard words appearing in less than 5 documents
# max_df=0.8, discard words appering in more than 80% of the documents
# sublinear_tf=True, use sublinear weighting
# use_idf=True, enable IDF
vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))

live_data = []
# negative
live_data.append("From what I understand this game was a kickstarter project which fine because most games from Kickstarters are usually good but I'm disappointed and so are the people that backed this obviously. Something went very wrong here during development and I feel like there wasn't a whole lot of quality testing performed on this game, I'll explain why I think that.")
# positive
live_data.append("I've been playing for a little bit now and I absolutely love this game. I could see how it's not for everyone but, I'm in love. I haven't run into any bugs yet either. I'm not usually very interested in the story of a game, but, I'm so into this story and setting. I hope that the issues that others are pointing out get addressed so that this game can continue to grow.")
# positive
live_data.append("I went into this expecting an okay game with a fantastic story, interesting mechanics, and a ton of glitches. What I got was a great game with a fantastic story, interesting mechanics, and an average amount of glitches. This game isn't as broken as people say. The world is easily the best part, but the stealth mechanic is strong on its own. They even gave Joy a system that's more than 'Don't take it'. You actually have to consider when it's necessary to use Joy and when you can get away without it, while also worrying about overdose/withdrawal.")
# negative
live_data.append("We Happy Few is a game a REALLY wanted to like, It's one I've been excited for since it's reveal with promise of it being a spiritual successor to Bioshock. Unfortunately it wasn't, In fact the game is barely playable at all. It's full of bugs and glitches at every turn, and I'm not normally one to let these things get to me, but it was so game breaking. ")
# negative
live_data.append("Complete garbage, slower than dirt. 1 month into using and its all messed up.")
# negative
live_data.append("Is a crappy computer as soon as i got it, it had a screen that that said it needed to be repair, mailed the seller and they have not responded")
# positive
live_data.append("This product is the perfect tool if you need something quick, durable and on the go! I will buy a memory card and/or an external hard drive as well. This is the tool i use when i am moving around the office or the house working and cant sit down. Power life is great. take your time with this product before giving a bad review. FOLLOW THE SET UP STEPS !!!!")

print(classifier_liblinear.predict(vectorizer.transform(live_data)))