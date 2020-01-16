import sys
import numpy as np
from scipy.io import loadmat
from processEmail import processEmail
from emailFeatures import emailFeatures
from svmTrain import svmTrain
from getVocabList import getVocabList

#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# ==================== Part 1: Email Preprocessing ====================

# Extract Features
with open('data/emailSample1.txt', 'r') as file:
	file_contents = file.read()
file.close()
#print(file_contents)
word_indices  = processEmail(file_contents)
#print(word_indices)
features = emailFeatures(word_indices)
print('Length of feature vector: ', len(features))
print('Number of non-zero entries: ', np.sum(features > 0))

# =========== Part 3: Train Linear SVM for Spam Classification ========

data = loadmat("data/spamTrain.mat")
X = data['X']
y = data['y']

C = 0.1
model = svmTrain(X, y, C, 'linear')
p = model.predict(X)
print('Training Accuracy: ', np.mean(np.double(p == y.ravel())) * 100)

# =================== Part 4: Test Spam Classification ================

data = loadmat("data/spamTest.mat")
Xtest = data['Xtest']
ytest = data['ytest']
p = model.predict(Xtest)
print('Test Accuracy: ', np.mean(np.double(p == ytest.ravel())) * 100)

# ================= Part 5: Top Predictors of Spam ====================

w = model.coef_[0]
idx = np.argsort(w)[::-1][:15]
vocabList = list(getVocabList().keys())
print('Top predictors of spam: ')
for i in idx:
	print("{:15s} {:.3f}".format(vocabList[i], w[i]))

# =================== Part 6: Try Your Own Emails =====================

with open('data/emailSample1.txt', 'r') as file:
	file_contents = file.read()
file.close()
word_indices  = processEmail(file_contents)
x = emailFeatures(word_indices)
p = model.predict(x.T)
print('Spam Classification: ', p)
print('(1 indicates spam, 0 indicates not spam)')
